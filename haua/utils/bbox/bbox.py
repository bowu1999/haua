from typing import List, Tuple, Optional, Literal, Union

import os
import cv2
import matplotlib.pyplot as plt


BoxFormat = Literal[
    'xyxy',
    'xyxyn',
    'xywh',
    'xywhn',
    'cxcywh',
    'cxcywhn']


def convert_box(
    box: Union[List[float], Tuple[float, ...]],
    fmt: BoxFormat,
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int]:
    if fmt == 'xyxy':
        x1, y1, x2, y2 = box

    elif fmt == 'xywh':
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h

    elif fmt == 'xyxyn':
        x1n, y1n, x2n, y2n = box
        x1, y1 = x1n * img_w, y1n * img_h
        x2, y2 = x2n * img_w, y2n * img_h

    elif fmt == 'xywhn':
        xn, yn, wn, hn = box
        x1, y1 = xn * img_w, yn * img_h
        x2, y2 = (xn + wn) * img_w, (yn + hn) * img_h

    elif fmt == 'cxcywh':
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

    elif fmt == 'cxcywhn':
        cxn, cyn, wn, hn = box
        cx = cxn * img_w
        cy = cyn * img_h
        w = wn * img_w
        h = hn * img_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w - 1))
    y2 = max(0, min(int(round(y2)), img_h - 1))

    return x1, y1, x2, y2


def draw_boxes_on_image(
    img_path: str,
    boxes: List[Union[List[float], Tuple[float, ...]]],
    fmt: BoxFormat = 'xyxy',
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show: bool = True,
    backend: Literal['cv2', 'plt'] = 'plt',  # 新增：默认用 matplotlib
    save_path: Optional[str] = None,
) -> str:
    """
    在图像上绘制目标框，并可选显示/保存。
    backend:
        - 'cv2': 使用 cv2.imshow（本地有 GUI 时用）
        - 'plt': 使用 matplotlib 显示（无 GUI / notebook 环境用）
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    img_h, img_w = img.shape[:2]

    if labels is not None and len(labels) != len(boxes):
        raise ValueError("labels length must match boxes length")

    if colors is None:
        colors = []
        for i in range(len(boxes)):
            color = (int(37 * i) % 255, int(17 * i) % 255, int(29 * i) % 255)
            colors.append(color)
    elif len(colors) != len(boxes):
        raise ValueError("colors length must match boxes length")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = convert_box(box, fmt, img_w, img_h)
        color = colors[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness)

        if labels is not None:
            label = str(labels[i])
            ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    if save_path is None:
        base, ext = os.path.splitext(img_path)
        save_path = base + "_boxed" + ext

    if save_path:
        cv2.imwrite(save_path, img)

    if show:
        if backend == 'cv2':
            # 只在本地有 GUI 的环境用
            cv2.imshow("Image with boxes", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif backend == 'plt':
            # matplotlib 显示（BGR -> RGB）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    return save_path