# 模型评估模块
## 目标检测评估
- run_onnx_inference
通过 `onnx` 模型生成推理结果，在指定路径下保存 predictions 文件
example：

```python
run_onnx_inference(
    ".../detection_model.onnx",
    ".../img_root/",
    ".../coco_style_annotation.json",
    ".../result_save_path/") # 可选参数
```

- calculate_coco_metrics
通过标注文件和推理结果计算 `coco` 指标

example:
```python
calculate_coco_metrics(
    ".../coco_style_annotation.json",
    ".../predictions.json")
```