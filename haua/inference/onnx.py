import json
from pathlib import Path
import numpy as np
import time
import onnxruntime as ort
# 假设 preprocess_image, draw_boxes 等函数已经定义


def run_onnx_inference(model_path: str, input_dir: str, output_dir: str, annotations_file: str):
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
        img_np, orig_size, orig_img = preprocess_image(img_path)
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
                "category_id": int(label),
                "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                "score": float(score)}
            predictions.append(prediction)

    predictions_output = output_dir / 'predictions.json'
    with open(predictions_output, 'w') as f:
        json.dump(predictions, f)