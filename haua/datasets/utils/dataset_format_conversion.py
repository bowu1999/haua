from typing import Optional, Tuple

import os
import json
import glob
import random
from PIL import Image



def yolo2coco(image_dir, label_dir, class_names, output_json):
    """
    将 YOLO 格式的标注转换为 COCO 格式。

    Args:
        image_dir (str): 图像文件夹路径（支持 .jpg, .png 等常见格式）
        label_dir (str): YOLO 标注文件夹路径（.txt 文件）
        class_names (list of str): 类别名称列表，顺序对应 YOLO 中的类别 ID
        output_json (str): 输出的 COCO JSON 文件路径
    """
    # 支持的图像扩展名
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    # 按文件名排序，确保一致性
    image_paths = sorted(image_paths)

    coco = {
        "images": [],
        "annotations": [],
        "categories": []}

    # 添加 categories，并确保类别ID从1开始
    for i, name in enumerate(class_names, start=1):  # 注意这里的start=1
        coco["categories"].append({
            "id": i,
            "name": name,
            "supercategory": "none"})

    annotation_id = 1

    for img_id, img_path in enumerate(image_paths, start=1):
        file_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(file_name)[0]
        label_path = os.path.join(label_dir, name_no_ext + '.txt')

        # 获取图像尺寸
        with Image.open(img_path) as img:
            width, height = img.size

        # 添加 image 信息
        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height})

        # 如果没有对应的 label 文件，跳过
        if not os.path.exists(label_path):
            continue

        # 读取 YOLO 标注
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 跳过无效行

            class_id = int(parts[0]) + 1  # YOLO类别ID从0开始，这里+1以匹配COCO的类别ID
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])

            # 转换为绝对坐标（像素）
            abs_x_center = x_center * width
            abs_y_center = y_center * height
            abs_bbox_width = bbox_width * width
            abs_bbox_height = bbox_height * height

            # 计算 COCO 格式的 (x_min, y_min, width, height)
            x_min = abs_x_center - abs_bbox_width / 2
            y_min = abs_y_center - abs_bbox_height / 2

            # 确保边界框不超出图像范围（可选）
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            abs_bbox_width = min(abs_bbox_width, width - x_min)
            abs_bbox_height = min(abs_bbox_height, height - y_min)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": class_id,  # 使用调整后的类别ID
                "bbox": [round(x_min, 2), round(y_min, 2), round(abs_bbox_width, 2), round(abs_bbox_height, 2)],
                "area": round(abs_bbox_width * abs_bbox_height, 2),
                "iscrowd": 0,
                "segmentation": []})
            annotation_id += 1

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(coco, f, indent=2)
        print(f"COCO 格式标注已保存至: {output_json}")

    return coco


def split_coco_dataset(
    coco_json_path: str,
    split_ratio: float = 0.8,
    output_dir: Optional[str] = None,
    train_json: Optional[str] = None,
    val_json: Optional[str] = None,
    seed: int = 42
) -> Tuple[dict, dict]:
    """
    将 COCO 格式数据集按图像级别划分为训练集和验证集。

    Args:
        coco_json_path (str): 输入的 COCO JSON 文件路径。
        split_ratio (float): 训练集占比（0.0 ~ 1.0），默认 0.8。
        output_dir (str, optional): 输出目录。若提供，则保存为 train.json / val.json。
        train_json (str, optional): 指定训练集输出路径（优先级高于 output_dir）。
        val_json (str, optional): 指定验证集输出路径（优先级高于 output_dir）。
        seed (int): 随机种子，确保可复现。

    Returns:
        tuple: (train_coco_dict, val_coco_dict)
    """
    assert 0.0 < split_ratio < 1.0, "split_ratio 必须在 (0, 1) 之间"

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    info = coco_data.get('info', {})
    licenses = coco_data.get('licenses', [])

    # 建立 image_id -> image 和 image_id -> [annotations]
    image_dict = {img['id']: img for img in images}
    anns_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # 获取所有 image_id 并打乱
    image_ids = list(image_dict.keys())
    random.seed(seed)
    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    def build_coco_subset(img_ids):
        subset_images = [image_dict[i] for i in img_ids]
        subset_anns = []
        for i in img_ids:
            subset_anns.extend(anns_by_image.get(i, []))
        return {
            "images": subset_images,
            "annotations": subset_anns,
            "categories": categories,
            "info": info,
            "licenses": licenses
        }

    train_coco = build_coco_subset(train_ids)
    val_coco = build_coco_subset(val_ids)

    # 确定输出路径
    input_dir = os.path.dirname(os.path.abspath(coco_json_path))

    if train_json is None or val_json is None:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_json = os.path.join(output_dir, 'train.json')
            val_json = os.path.join(output_dir, 'val.json')
        else:
            train_json = os.path.join(input_dir, 'train.json')
            val_json = os.path.join(input_dir, 'val.json')

    # 保存文件
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2, ensure_ascii=False)
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, indent=2, ensure_ascii=False)

    print(f"训练集已保存至: {train_json} （{len(train_coco['images'])} 张图像）")
    print(f"验证集已保存至: {val_json} （{len(val_coco['images'])} 张图像）")

    return train_coco, val_coco