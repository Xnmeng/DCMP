import os
import csv
import shutil

# ===== 改成你的实际路径 =====
root = "/home/lc/GCD_datasets/imagenet"
val_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC/val")
csv_file = os.path.join(root, "LOC_val_solution.csv")

# 是否保留原图（True=复制，False=移动）
copy_instead_of_move = False

with open(csv_file, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        if len(row) < 2:
            continue

        image_id = row[0].strip()
        pred_str = row[1].strip()

        if not image_id or not pred_str:
            continue

        # PredictionString 通常形如: "n01751748 1.0"
        wnid = pred_str.split()[0]

        src_img = os.path.join(val_dir, image_id + ".JPEG")
        dst_dir = os.path.join(val_dir, wnid)
        dst_img = os.path.join(dst_dir, image_id + ".JPEG")

        if not os.path.exists(src_img):
            print(f"[跳过] 找不到图片: {src_img}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        if copy_instead_of_move:
            shutil.copy2(src_img, dst_img)
        else:
            shutil.move(src_img, dst_img)

print("处理完成：val 已整理为 ImageFolder 格式。")