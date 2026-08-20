import math
import copy
import torch
import torch.nn as nn
from clip import clip
from torch.optim.lr_scheduler import _LRScheduler


class Adapter(nn.Module):
    """CoPrompt adapter on top of an embedding branch."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PromptedTextEncoder(nn.Module):
    """Sample-level prompted text encoder for OWSSL.

    Original CoPrompt builds learnable prompts around closed-set class names.
    In OWSSL/GCD we do not know the names of the novel classes, so we keep the
    *mechanism* of CoPrompt (shared shallow prompts + deeper coupled prompts)
    but apply it to sample-level RTG text tokens.
    """

    def __init__(self, clip_model, n_ctx: int):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.context_length = clip_model.context_length
        self.n_ctx = n_ctx

    def forward(self, tokenized_text, shared_ctx_text, deep_text_prompts):
        x = self.token_embedding(tokenized_text).type(self.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1 : self.context_length - self.n_ctx, :]
        ctx = shared_ctx_text.unsqueeze(0).expand(x.shape[0], -1, -1).type(self.dtype)
        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        outputs = self.transformer([x, deep_text_prompts, 0])
        x = outputs[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        eot_positions = tokenized_text.argmax(dim=-1) + self.n_ctx
        eot_positions = torch.clamp(eot_positions, max=x.shape[1] - 1)
        x = x[torch.arange(x.shape[0]), eot_positions] @ self.text_projection
        return x


class SharedPromptLearner(nn.Module):
    """Coupled prompt learner in the spirit of CoPrompt/MaPLe.

    One shallow prompt is learned in the text space and projected to the visual
    width. Deeper prompts are also learned in text space and projected to the
    visual branch layer by layer.
    """

    def __init__(self, clip_model, n_ctx: int = 4, prompt_depth: int = 12):
        super().__init__()
        text_dim = clip_model.ln_final.weight.shape[0]
        if not hasattr(clip_model.visual, "conv1"):
            raise ValueError("FullCoPromptOWSSL currently supports ViT-based CLIP backbones only.")
        vision_width = clip_model.visual.conv1.weight.shape[0]

        self.n_ctx = n_ctx
        self.prompt_depth = prompt_depth
        self.ctx = nn.Parameter(torch.empty(n_ctx, text_dim, dtype=clip_model.dtype))
        nn.init.normal_(self.ctx, std=0.02)

        self.ctx_proj = nn.Linear(text_dim, vision_width)

        num_deep = max(prompt_depth - 1, 0)
        self.deep_text_prompts = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, text_dim, dtype=clip_model.dtype)) for _ in range(num_deep)]
        )
        for p in self.deep_text_prompts:
            nn.init.normal_(p, std=0.02)

        layer = nn.Linear(text_dim, vision_width)
        self.deep_prompt_projections = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_deep)])

    def forward(self):
        shared_ctx_text = self.ctx
        shared_ctx_vision = self.ctx_proj(self.ctx).type(self.ctx_proj.weight.dtype)

        deep_text = list(self.deep_text_prompts)
        deep_vision = []
        for proj, prompt in zip(self.deep_prompt_projections, self.deep_text_prompts):
            deep_vision.append(proj(prompt).type(proj.weight.dtype))

        return shared_ctx_text, shared_ctx_vision, deep_text, deep_vision


class CustomCLIP(nn.Module):
    def __init__(self, clip_model, class_nums):
        super().__init__()
        self.model = clip_model
        self.outputdim = clip_model.visual.output_dim

        self.image_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.image_classifier.weight_g.data.fill_(1)
        self.image_classifier.weight_g.requires_grad = False

        self.text_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.text_classifier.weight_g.data.fill_(1)
        self.text_classifier.weight_g.requires_grad = False

    def encode_image(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, tokens):
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)

        logits_image = self.image_classifier(image_features)
        logits_text = self.text_classifier(text_features)

        return logits_image, logits_text, image_features, text_features


class FullCoPromptOWSSL(nn.Module):
    """CoPrompt-style OWSSL model.

    Important note:
    - The *mechanism* follows CoPrompt closely: coupled multi-modal prompts,
      adapters on both branches, and consistency against frozen CLIP features on
      perturbed image/text inputs.
    - The *task head* follows TextGCD/OWSSL: two parametric classifiers and the
      three-stage warm-up / class-align / co-teaching schedule.

    This is necessary because the original closed-set CoPrompt assumes that all
    class names are known, which is incompatible with OWSSL novel classes.
    """

    def __init__(
        self,
        clip_model,
        class_nums,
        n_ctx=4,
        prompt_depth=12,
        image_adapter_reduction=4,
        text_adapter_reduction=4,
        image_adapter_m=0.1,
        text_adapter_m=0.2,
    ):
        super().__init__()
        self.model = clip_model
        self.outputdim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        self.n_ctx = n_ctx

        self.prompt_learner = SharedPromptLearner(clip_model, n_ctx=n_ctx, prompt_depth=prompt_depth)
        self.prompted_text_encoder = PromptedTextEncoder(clip_model, n_ctx=n_ctx)
        self.image_encoder = clip_model.visual

        self.image_adapter = Adapter(self.outputdim, reduction=image_adapter_reduction)
        self.text_adapter = Adapter(self.outputdim, reduction=text_adapter_reduction)
        self.image_adapter_m = image_adapter_m
        self.text_adapter_m = text_adapter_m

        self.image_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.image_classifier.weight_g.data.fill_(1)
        self.image_classifier.weight_g.requires_grad = False

        self.text_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.text_classifier.weight_g.data.fill_(1)
        self.text_classifier.weight_g.requires_grad = False

    def encode_image_prompted(self, image):
        _, shared_ctx_vision, _, deep_vision_prompts = self.prompt_learner()
        feats = self.image_encoder(image.type(self.dtype), shared_ctx_vision, deep_vision_prompts)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text_prompted(self, tokens):
        shared_ctx_text, _, deep_text_prompts, _ = self.prompt_learner()
        feats = self.prompted_text_encoder(tokens, shared_ctx_text, deep_text_prompts)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_image_frozen(self, image):
        feats = self.model.encode_image(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text_frozen(self, tokens):
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def adapt_image(self, feats):
        delta = self.image_adapter(feats)
        mixed = self.image_adapter_m * delta + (1 - self.image_adapter_m) * feats
        mixed = mixed / mixed.norm(dim=-1, keepdim=True)
        return mixed

    def adapt_text(self, feats):
        delta = self.text_adapter(feats)
        mixed = self.text_adapter_m * delta + (1 - self.text_adapter_m) * feats
        mixed = mixed / mixed.norm(dim=-1, keepdim=True)
        return mixed

    def forward(self, images, text, frozen_images=None, frozen_text=None, return_aux=False):
        raw_image_features = self.encode_image_prompted(images)
        raw_text_features = self.encode_text_prompted(text)

        image_features = self.adapt_image(raw_image_features)
        text_features = self.adapt_text(raw_text_features)

        logits_image = self.image_classifier(image_features)
        logits_text = self.text_classifier(text_features)

        if not return_aux:
            return logits_image, logits_text, image_features, text_features

        aux = {
            "raw_image_features": raw_image_features,
            "raw_text_features": raw_text_features,
            "adapted_image_features": image_features,
            "adapted_text_features": text_features,
        }

        if frozen_images is not None:
            aux["frozen_image_features"] = self.encode_image_frozen(frozen_images)
        if frozen_text is not None:
            aux["frozen_text_features"] = self.encode_text_frozen(frozen_text)

        return logits_image, logits_text, image_features, text_features, aux


class CustomCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, classifier_params, T_max, eta_min=0, last_epoch=-1):
        self.classifier_params_ids = set(map(id, classifier_params))
        self.T_max = T_max
        self.eta_min = eta_min
        self.classifier_lr = optimizer.param_groups[0]['lr']
        self.base_lr = optimizer.param_groups[1]['lr']
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (self.classifier_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            if any(id(p) in self.classifier_params_ids for p in param_group['params'])
            else self.base_lr
            for param_group in self.optimizer.param_groups
        ]


class ImageViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for _ in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


class TextViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        if not isinstance(base_transform, list):
            if not callable(base_transform):
                raise ValueError("The text transformation must be callable.")
        else:
            if not all(callable(f) for f in base_transform):
                raise ValueError("All elements in the text transformations list must be callable.")
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, text):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(text) for _ in range(self.n_views)]
        else:
            return [self.base_transform[i](text) for i in range(self.n_views)]


def _load_state_dict(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return state_dict or model.state_dict()


def load_clip_to_cpu(backbone_name):
    state_dict = _load_state_dict(backbone_name)
    design_details = {
        "trainer": "CoOp",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": 0,
    }
    model = clip.build_model(state_dict, design_details)
    return model


def load_prompt_clip_to_cpu(backbone_name, n_ctx=4, prompt_depth=12):
    state_dict = _load_state_dict(backbone_name)
    design_details = {
        "trainer": "CoPrompt",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": n_ctx,
    }
    model = clip.build_model(state_dict, design_details)
    return model
