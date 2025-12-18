import json
import time
import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _preprocess_image(image_path, target_size=(640, 640)):
    """
    读取图像并转换为模型输入
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(target_size)
    img_np = np.array(img_resized).astype(np.float32)
    # HWC -> CHW
    img_np = img_np.transpose(2, 0, 1) / 255.0
    img_np = np.expand_dims(img_np, axis=0)  # 增加 batch 维度
    # orig_target_sizes 必须是 int64
    orig_size = np.array([[img.height, img.width]], dtype=np.int64)

    return img_np, orig_size, img


def run_onnx_inference(model_path: str, input_dir: str, annotations_file: str, output_dir: str = None):
    model_path = Path(model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]

    with open(annotations_file, 'r') as f:
        dataset = json.load(f)

    predictions = []

    for img_info in dataset['images']:
        img_path = input_dir / img_info['file_name']
        if not img_path.exists():
            continue
        img_np, orig_size, orig_img = _preprocess_image(img_path)
        inputs = {
            input_names[0]: img_np,
            input_names[1]: np.array([[640, 640]], dtype=np.int64)}
        outputs = session.run(output_names, inputs)
        labels, boxes, scores = outputs

        # 处理输出...
        labels = np.array(labels).reshape(-1)
        scores = np.array(scores).reshape(-1)
        boxes = np.array(boxes).reshape(-1, 4)

        mask = scores >= 0.6
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        h_ratio = orig_size[0,0] / 640
        w_ratio = orig_size[0,1] / 640
        boxes[:, [0,2]] *= w_ratio
        boxes[:, [1,3]] *= h_ratio

        for label, box, score in zip(labels, boxes, scores):
            prediction = {
                "image_id": img_info['id'],
                "category_id": int(label) + 1,
                "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                "score": float(score)}
            predictions.append(prediction)
    if output_dir:
        predictions_output = output_dir / 'predictions.json'
        with open(predictions_output, 'w') as f:
            json.dump(predictions, f)
    
    return predictions


def calculate_coco_metrics(annotations_file: str, predictions_file: str):
    cocoGt = COCO(annotations_file)  # 加载标注文件
    cocoDt = cocoGt.loadRes(predictions_file)  # 加载预测文件

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats