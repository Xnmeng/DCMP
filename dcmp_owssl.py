import math
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.get_datasets import get_datasets, get_class_splits
from data.augmentations import (
    get_transform,
    build_coprompt_prompt_text,
    build_coprompt_rich_text,
)
from model import (
    FullCoPromptOWSSL,
    CustomCosineAnnealingLR,
    ImageViewGenerator,
    TextViewGenerator,
    load_prompt_clip_to_cpu,
)
from utils import (
    init_experiment,
    select_confident_samples,
    image_text_contrastive_loss,
    simgcd_loss,
    coteaching_pseudolabel_loss,
    evaluate_accuracy,
    feature_consistency_loss,
)

from config import cub_root, cub_retrieved_text_path, flowers_root, flowers_retrieved_text_path, scars_root, scars_retrieved_text_path, \
                   pets_root, pets_retrieved_text_path, cifar10_root, cifar10_retrieved_text_path, cifar100_root, cifar100_retrieved_text_path, imagenet_root, imagenet_retrieved_text_path, osr_split_dir


def train_one_epoch(args, logger, writer, loader, model, optimizer, epoch, selected_samples_t, selected_samples_i):
    model.train()
    adjust_learning_rate(optimizer, epoch, args)
    total_loss = 0.0
    total_loss_base = 0.0
    total_loss_con = 0.0
    total_loss_pseudo_image = 0.0
    total_loss_pseudo_text = 0.0
    total_loss_lcc_image = 0.0
    total_loss_lcc_text = 0.0

    teacher_temp_schedule = np.concatenate((
        np.linspace(args.tau_t_start, args.tau_t_end, args.warmup_teacher_temp_epochs),
        np.ones(args.epochs - args.warmup_teacher_temp_epochs) * args.tau_t_end
    ))

    param_group_names = ['classifier_head', 'prompt_and_adapter']

    for batch_idx, (images, labels, img_id, descriptive_text_token, _, mask) in enumerate(tqdm(loader, desc="Training")):
        mask = mask[:, 0]
        labels = labels.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True).bool()

        image_views = [v.cuda(non_blocking=True) for v in images]
        text_views = [torch.squeeze(v, 1).cuda(non_blocking=True) for v in descriptive_text_token]

        images_cat = torch.cat(image_views, dim=0)
        text_cat = torch.cat(text_views, dim=0)

        frozen_images = image_views[args.frozen_image_view_idx]
        frozen_text = text_views[args.frozen_text_view_idx]

        logits_image, logits_text, image_features, text_features, aux = model(
            images_cat,
            text_cat,
            frozen_images=frozen_images,
            frozen_text=frozen_text,
            return_aux=True,
        )

        loss_con = image_text_contrastive_loss(image_features, text_features, model.model.logit_scale.exp(), args)
        loss_base = simgcd_loss(logits_image, logits_text, labels, mask, teacher_temp_schedule, epoch, args)
        loss_pseudo_image = coteaching_pseudolabel_loss(selected_samples_t, logits_image, img_id, args)
        loss_pseudo_text = coteaching_pseudolabel_loss(selected_samples_i, logits_text, img_id, args)

        prompted_image_cc = aux["adapted_image_features"].chunk(args.n_views)[args.prompt_image_view_idx]
        prompted_text_cc = aux["adapted_text_features"].chunk(args.n_views)[args.prompt_text_view_idx]
        frozen_image_features = aux["frozen_image_features"]
        frozen_text_features = aux["frozen_text_features"]

        loss_lcc_image = feature_consistency_loss(prompted_image_cc, frozen_image_features, criterion="cosine")
        loss_lcc_text = feature_consistency_loss(prompted_text_cc, frozen_text_features, criterion="cosine")

        loss = (
            loss_base
            + args.lambda_contrast * loss_con
            + loss_pseudo_image
            + loss_pseudo_text
            + args.lambda_lcc * (loss_lcc_image + loss_lcc_text)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_base += loss_base.item()
        total_loss_con += loss_con.item()
        total_loss_pseudo_image += loss_pseudo_image.item()
        total_loss_pseudo_text += loss_pseudo_text.item()
        total_loss_lcc_image += loss_lcc_image.item()
        total_loss_lcc_text += loss_lcc_text.item()

        iter_idx = epoch * len(loader) + batch_idx
        writer.add_scalars('Loss', {
            'loss_base': loss_base.item(),
            'loss_con': loss_con.item(),
            'loss_pseudo_image': loss_pseudo_image.item(),
            'loss_pseudo_text': loss_pseudo_text.item(),
            'loss_lcc_image': loss_lcc_image.item(),
            'loss_lcc_text': loss_lcc_text.item(),
            'total_loss': loss.item()
        }, iter_idx)

    #scheduler.step()
    logger.info(
        f"Epoch {epoch+1}/{args.epochs}, Total Loss: {total_loss / len(loader):.4f}, "
        f"Base Loss: {total_loss_base / len(loader):.4f}, Con Loss: {total_loss_con / len(loader):.4f}, "
        f"Pseudo Loss Image: {total_loss_pseudo_image / len(loader):.4f}, Pseudo Loss Text: {total_loss_pseudo_text / len(loader):.4f}, "
        f"Lcc Image: {total_loss_lcc_image / len(loader):.4f}, Lcc Text: {total_loss_lcc_text / len(loader):.4f}"
    )
    for idx, param_group in enumerate(optimizer.param_groups):
        logger.info(f"   Param Group: {param_group_names[idx]}, Learning Rate: {param_group['lr']:.6f}")


def _predict_from_mode(logits_image, logits_text, args):
    image_probs = F.softmax(logits_image, dim=-1)
    text_probs = F.softmax(logits_text, dim=-1)
    return image_probs + text_probs


def adjust_learning_rate(optimizer, epoch, args):
    """
    param_groups[0] -> classifier_head
    param_groups[1] -> prompt_and_adapter
    """
    classifier_max_lr = args.classifier_lr
    classifier_min_lr = args.classifier_lr * 1e-3   # 和你原来一致：0.1 -> 1e-4

    prompt_max_lr = args.base_lr
    prompt_min_lr = args.prompt_min_lr              # 新增：5e-4 -> 5e-5

    # 让最后一轮更接近最小 lr
    denom = max(1, args.epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / denom))

    optimizer.param_groups[0]['lr'] = classifier_min_lr + (classifier_max_lr - classifier_min_lr) * cosine
    optimizer.param_groups[1]['lr'] = prompt_min_lr + (prompt_max_lr - prompt_min_lr) * cosine


def test(model, test_loader, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for _, (images, label, _, descriptive_text_token, _) in enumerate(tqdm(test_loader, desc="Testing")):
        images = images.cuda(non_blocking=True)
        descriptive_text_token = descriptive_text_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, descriptive_text_token)
            probs = _predict_from_mode(logits_image, logits_text, args)
            preds.append(probs.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    total_acc, old_acc, new_acc = evaluate_accuracy(preds, targets, mask)
    return total_acc, old_acc, new_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full CoPrompt-style OWSSL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_dir', default='exp', type=str)
    parser.add_argument('--experiment_name', default='cub_full_coprompt_owssl', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--dataset_name', default='cub', type=str, help='options: cifar10, cifar100, imagenet_100, cub, scars, pets, flowers')
    parser.add_argument('--backbone_name', default='ViT-B/16', type=str, help="CLIP backbone")

    parser.add_argument('--cub_root', default=cub_root, type=str)
    parser.add_argument('--cub_retrieved_text_path', default=cub_retrieved_text_path, type=str)
    parser.add_argument('--flowers_root', default=flowers_root, type=str)
    parser.add_argument('--flowers_retrieved_text_path', default=flowers_retrieved_text_path, type=str)
    parser.add_argument('--scars_root', default=scars_root, type=str)
    parser.add_argument('--scars_retrieved_text_path', default=scars_retrieved_text_path, type=str)
    parser.add_argument('--pets_root', default=pets_root, type=str)
    parser.add_argument('--pets_retrieved_text_path', default=pets_retrieved_text_path, type=str)
    parser.add_argument('--cifar10_root', default=cifar10_root, type=str)
    parser.add_argument('--cifar10_retrieved_text_path', default=cifar10_retrieved_text_path, type=str)
    parser.add_argument('--cifar100_root', default=cifar100_root, type=str)
    parser.add_argument('--cifar100_retrieved_text_path', default=cifar100_retrieved_text_path, type=str)
    parser.add_argument('--imagenet_root', default=imagenet_root, type=str)
    parser.add_argument('--imagenet_retrieved_text_path', default=imagenet_retrieved_text_path, type=str)
    parser.add_argument('--osr_split_dir', type=str, default=osr_split_dir)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--base_lr', default=0.0005, type=float)
    parser.add_argument('--classifier_lr', default=0.1, type=float)
    parser.add_argument('--prompt_min_lr', default=5e-5, type=float,
                    help='minimum lr for prompt/adapter param group')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--prop_train_labels', default=0.5, type=float)
    parser.add_argument('--use_ssb_splits', default=True, action='store_true')

    parser.add_argument('--transform', default='imagenet', type=str)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--prompt_image_view_idx', default=0, type=int)
    parser.add_argument('--frozen_image_view_idx', default=1, type=int)
    parser.add_argument('--prompt_text_view_idx', default=0, type=int)
    parser.add_argument('--frozen_text_view_idx', default=1, type=int)

    parser.add_argument('--selecting_ratio', default=0.6, type=float)
    parser.add_argument('--lambda_loss', default=0.2, type=float)
    parser.add_argument('--warm_up_epochs', default=10, type=int)
    parser.add_argument('--class_aligning_epochs', default=5, type=int)

    parser.add_argument('--num_attributes', default=2, type=int)
    parser.add_argument('--num_tags', default=3, type=int)

    parser.add_argument('--tau_s', default=0.1, type=float)
    parser.add_argument('--tau_u', default=0.05, type=float)
    parser.add_argument('--tau_t_start', default=0.035, type=float)
    parser.add_argument('--tau_t_end', default=0.02, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int)
    parser.add_argument('--memax_weight', default=2, type=float)

    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--prompt_depth', default=12, type=int)
    parser.add_argument('--image_adapter_reduction', default=4, type=int)
    parser.add_argument('--text_adapter_reduction', default=4, type=int)
    parser.add_argument('--image_adapter_m', default=0.1, type=float)
    parser.add_argument('--text_adapter_m', default=0.2, type=float)
    parser.add_argument('--lambda_contrast', default=1.0, type=float)
    parser.add_argument('--lambda_lcc', default=4.0, type=float)

    parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='only run evaluation')

    args, ignored_args = parser.parse_known_args()
    if ignored_args:
        print(f"Ignoring unsupported command-line arguments: {ignored_args}")
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args, logger, writer = init_experiment(args)

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    logger.info(f"Loading prompt-capable CLIP (backbone: {args.backbone_name})")
    backbone = load_prompt_clip_to_cpu(args.backbone_name, n_ctx=args.n_ctx, prompt_depth=args.prompt_depth).float()

    logger.info("Building full CoPrompt OWSSL model")
    model = FullCoPromptOWSSL(
        backbone,
        args.mlp_out_dim,
        n_ctx=args.n_ctx,
        prompt_depth=args.prompt_depth,
        image_adapter_reduction=args.image_adapter_reduction,
        text_adapter_reduction=args.text_adapter_reduction,
        image_adapter_m=args.image_adapter_m,
        text_adapter_m=args.text_adapter_m,
    ).to(args.device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {args.resume}")

    logger.info("Turning off gradients in encoders and keeping only prompts/adapters/classifiers trainable")
    for _, param in model.named_parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if (
            "prompt_learner" in name
            or "image_adapter" in name
            or "text_adapter" in name
            or "image_classifier" in name
            or "text_classifier" in name
        ):
            param.requires_grad_(True)

    params_names = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info("Parameters that require gradients: %s", params_names)

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ImageViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # CoPrompt-style text pair:
    #   view-0: concise prompt-like text
    #   view-1: richer RTG description for frozen consistency target
    text_transform = TextViewGenerator(
        base_transform=[build_coprompt_prompt_text, build_coprompt_rich_text],
        n_views=args.n_views,
    )

    train_dataset, test_dataset, unlabelled_train_examples_test, _ = get_datasets(
        args.dataset_name, train_transform, test_transform, text_transform, args
    )
    logger.info(f"len of train dataset: {len(train_dataset)}")
    logger.info(f"len of test dataset: {len(unlabelled_train_examples_test)}")

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    if args.eval_only:
        total_acc, old_acc, new_acc = test(model, test_loader_unlabelled, args)
        logger.info(
            f"[Eval] All {total_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}"
        )
        exit(0)

    classifier_params_train = [p for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]
    classifier_params_train_name = [name for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]
    logger.info("Parameters in classifier with big lr: %s", classifier_params_train_name)

    other_params_train = [p for name, p in model.named_parameters() if "classifier" not in name and p.requires_grad]
    classifier_lr = args.classifier_lr
    base_lr = args.base_lr

 
    optimizer_train = SGD([
        {'params': classifier_params_train, 'lr': classifier_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay},
        {'params': other_params_train, 'lr': base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    ])

    '''
    scheduler_train = CustomCosineAnnealingLR(optimizer_train, classifier_params_train, T_max=args.epochs, eta_min=classifier_lr * 1e-3)
    '''


    selecting_nums = math.floor(len(test_loader_unlabelled.dataset) / args.mlp_out_dim * args.selecting_ratio)
    logger.info(f"Selecting {selecting_nums} high-confidence samples for co-teaching.")

    best_all_acc = 0.0
    save_dir = os.path.dirname(args.model_path)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(args.epochs):
        
        if epoch > args.class_aligning_epochs + args.warm_up_epochs:
            selected_samples_i = select_confident_samples(model, test_loader_unlabelled, selecting_nums)
            selected_samples_t = select_confident_samples(model, test_loader_unlabelled, selecting_nums, from_image=False)
       

        elif epoch > args.warm_up_epochs:
            selected_samples_i = None
            selected_samples_t = select_confident_samples(model, test_loader_unlabelled, selecting_nums, from_image=False)
        else:
            selected_samples_i = None
            selected_samples_t = None

        if selected_samples_i:
            logger.info(f"len of image selected samples: {len(selected_samples_i)}")
        if selected_samples_t:
            logger.info(f"len of text selected samples: {len(selected_samples_t)}")

        train_one_epoch(args, logger, writer, train_loader, model, optimizer_train,
                        epoch, selected_samples_t, selected_samples_i)
        total_acc, old_acc, new_acc = test(model, test_loader_unlabelled, args)
        logger.info(f"Weighted Accuracies: All {total_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}")

        writer.add_scalar('Accuracy/All', total_acc, epoch)
        writer.add_scalar('Accuracy/Old', old_acc, epoch)
        writer.add_scalar('Accuracy/New', new_acc, epoch)

        # 1) 每轮都保存 checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_train.state_dict(),
            'current_classifier_lr': optimizer_train.param_groups[0]['lr'],
            'current_prompt_lr': optimizer_train.param_groups[1]['lr'],
            'best_all_acc': best_all_acc,
            'all_acc': total_acc,
            'old_acc': old_acc,
            'new_acc': new_acc,
            'args': vars(args),
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # 2) 如果当前 All acc 更好，就保存 best_model
        if total_acc > best_all_acc:
            best_all_acc = total_acc

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_train.state_dict(),
                'current_classifier_lr': optimizer_train.param_groups[0]['lr'],
                'current_prompt_lr': optimizer_train.param_groups[1]['lr'],
                'best_all_acc': best_all_acc,
                'all_acc': total_acc,
                'old_acc': old_acc,
                'new_acc': new_acc,
                'args': vars(args),
            }
            torch.save(best_checkpoint, best_model_path)
            logger.info(
                f"Saved best model to {best_model_path} "
                f"(All {total_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f})"
            )

    writer.close()
