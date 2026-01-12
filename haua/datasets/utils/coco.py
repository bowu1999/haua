import copy
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional


def splitCOCO(
    src_json_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    save_json_path: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    将一个 COCO 标注 json 按图像维度划分为 train/val 两个子集。

    Args:
        src_json_path: 原始 COCO 标注文件路径
        train_json_path: 输出的 train COCO 标注文件路径
        val_json_path: 输出的 val COCO 标注文件路径
        val_ratio: 验证集比例，例如 0.2 表示 20% 图像划为 val
        seed: 随机种子，保证可复现划分结果
    
    Returns:
        tuple(train_coco_dict, val_coco_dict)
    """
    with open(src_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images: List[Dict[str, Any]] = coco.get("images", [])
    annotations: List[Dict[str, Any]] = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_ids = [img["id"] for img in images]

    random.seed(seed)
    random.shuffle(image_ids)

    num_val = int(len(image_ids) * val_ratio)
    val_image_ids = set(image_ids[:num_val])
    train_image_ids = set(image_ids[num_val:])

    train_images = [img for img in images if img["id"] in train_image_ids]
    val_images = [img for img in images if img["id"] in val_image_ids]

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    train_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories}
    val_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories}

    if save_json_path:
        Path(save_json_path).mkdir(parents=True, exist_ok=True)
        train_json_path = Path(save_json_path) / "train.json"
        val_json_path = Path(save_json_path) / "val.json"
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(train_coco, f, ensure_ascii=False, indent=2)
        with open(val_json_path, "w", encoding="utf-8") as f:
            json.dump(val_coco, f, ensure_ascii=False, indent=2)

    return train_coco, val_coco


def mergeCOCODatasets(coco1_path, coco2_path, output_path, category_mapping=None):
    """
    合并两个COCO格式的数据集。
    
    Args:
        coco1_path (str): 第一个数据集的json路径 (基准数据集)。
        coco2_path (str): 第二个数据集的json路径 (将被合并的数据集)。
        output_path (str): 输出合并后json的路径。
        category_mapping (list[int], optional): 
            一个列表，长度必须等于coco2中类别的数量。
            列表中的第 i 个元素代表 coco2 中第 i 个类别（按id排序）应该映射到 coco1 中的哪个 category_id。
            如果为 None，则默认 ID 不变 (1->1, 2->2...)。
    """
    print(f"正在加载数据集...")
    with open(coco1_path, 'r', encoding='utf-8') as f:
        coco1 = json.load(f)
    with open(coco2_path, 'r', encoding='utf-8') as f:
        coco2 = json.load(f)
    # info 和 licenses 字段保留在 coco1 中，此处不做修改，自然保留第一个数据集的信息
    # 处理类别映射 (Category Mapping)
    # 获取两个数据集的类别列表，并按ID排序确保顺序一致
    coco1_cats = sorted(coco1.get('categories', []), key=lambda x: x['id'])
    coco2_cats = sorted(coco2.get('categories', []), key=lambda x: x['id'])
    # 建立 coco1 现有的 ID 集合，用于判断是否需要新增类别
    coco1_cat_ids = {cat['id'] for cat in coco1_cats}
    # 建立 coco2 旧ID 到 新ID 的映射字典
    # 格式: {coco2_old_id: merged_new_id}
    id_map_2to1 = {}
    if category_mapping is None:
        print("未提供映射参数，默认使用原始ID合并...")
        # 如果没有提供映射，假设 ID 是一一对应的
        for cat in coco2_cats:
            old_id = cat['id']
            target_id = old_id
            id_map_2to1[old_id] = target_id
            # 如果这个ID在coco1里不存在，则添加进去
            if target_id not in coco1_cat_ids:
                new_cat = copy.deepcopy(cat)
                coco1['categories'].append(new_cat)
                coco1_cat_ids.add(target_id)
                print(f"  [新增类别] ID {target_id}: {cat['name']}")
    else:
        print(f"使用自定义映射参数: {category_mapping}")
        if len(category_mapping) != len(coco2_cats):
            raise ValueError(f"映射参数长度 ({len(category_mapping)}) 与 数据集2的类别数量 ({len(coco2_cats)}) 不匹配！")
        for idx, target_id in enumerate(category_mapping):
            source_cat = coco2_cats[idx]
            source_id = source_cat['id']
            # 记录映射关系：coco2的 source_id 变成 target_id
            id_map_2to1[source_id] = target_id
            # 逻辑：如果 target_id 已经在 coco1 中存在，则保留 coco1 的名字（不做操作）
            # 如果 target_id 不在 coco1 中，则将 coco2 的这个类添加进 coco1
            if target_id not in coco1_cat_ids:
                new_cat = copy.deepcopy(source_cat)
                new_cat['id'] = target_id
                # 名字沿用 coco2 的名字
                coco1['categories'].append(new_cat)
                coco1_cat_ids.add(target_id)
                print(f"  [新增类别] ID {target_id}: {source_cat['name']} (来自数据集2的 {source_cat['name']})")
            else:
                # 仅仅为了打印日志，找到对应的coco1类别名
                target_name = next(c['name'] for c in coco1['categories'] if c['id'] == target_id)
                print(f"  [合并类别] 数据集2 '{source_cat['name']}' (ID:{source_id}) -> 数据集1 '{target_name}' (ID:{target_id})")
    # 重新排序 categories 以保持整洁
    coco1['categories'].sort(key=lambda x: x['id'])
    # 处理图片 (Images)
    # 为了防止图片ID冲突，我们需要找到 coco1 中最大的 image_id
    max_img_id = 0
    if coco1.get('images'):
        max_img_id = max(img['id'] for img in coco1['images'])
    print(f"正在合并图片 (起始 ID: {max_img_id + 1})...")
    # 建立图片ID映射: {coco2_img_id: new_unique_img_id}
    img_id_map = {}
    for img in coco2.get('images', []):
        old_id = img['id']
        max_img_id += 1
        new_id = max_img_id
        img_id_map[old_id] = new_id
        new_img = copy.deepcopy(img)
        new_img['id'] = new_id
        coco1['images'].append(new_img)
    # 处理标注 (Annotations)
    # 同样需要防止 annotation ID 冲突
    max_ann_id = 0
    if coco1.get('annotations'):
        max_ann_id = max(ann['id'] for ann in coco1['annotations'])  
    print(f"正在合并标注 (起始 ID: {max_ann_id + 1})...")
    for ann in coco2.get('annotations', []):
        new_ann = copy.deepcopy(ann)
        # 更新 annotation id
        max_ann_id += 1
        new_ann['id'] = max_ann_id
        # 更新 image_id (使用上面的映射)
        if ann['image_id'] not in img_id_map:
            print(f"警告: 标注 {ann['id']} 对应的图片 {ann['image_id']} 在 images 列表中未找到，跳过。")
            continue
        new_ann['image_id'] = img_id_map[ann['image_id']]
        # 更新 category_id (使用上面的映射)
        if ann['category_id'] not in id_map_2to1:
             print(f"警告: 标注 {ann['id']} 的类别 ID {ann['category_id']} 不在类别列表中，跳过。")
             continue
        new_ann['category_id'] = id_map_2to1[ann['category_id']]
        coco1['annotations'].append(new_ann)
    # 保存结果
    print(f"保存合并结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco1, f, indent=None) # indent=None 减小文件体积，如需可读性可设为2
    print("完成！")