import cv2
import numpy as np


KPT_CONF_THRESHOLD = 0.5

# YOLO 姿态模型使用的 COCO 17 点骨架连接关系。
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

SKELETON_COLORS = {
    (0, 1): (255, 200, 0),
    (0, 2): (255, 200, 0),
    (1, 3): (255, 150, 0),
    (2, 4): (255, 150, 0),
    (5, 6): (255, 255, 255),
    (5, 11): (255, 255, 255),
    (6, 12): (255, 255, 255),
    (11, 12): (255, 255, 255),
    (5, 7): (0, 255, 80),
    (7, 9): (0, 200, 60),
    (6, 8): (0, 255, 200),
    (8, 10): (0, 200, 180),
    (11, 13): (0, 140, 255),
    (13, 15): (0, 100, 200),
    (12, 14): (180, 60, 255),
    (14, 16): (140, 40, 200),
}

KPT_COLOR_HEAD = (0, 255, 255)
KPT_COLOR_BODY = (0, 80, 255)


def draw_stick_figure(
    img: np.ndarray,
    keypoints: np.ndarray,
    kpt_conf: np.ndarray,
    threshold: float = KPT_CONF_THRESHOLD,
) -> None:
    # 先画肢体连线，确保关键点圆点始终显示在最上层。
    for i, j in SKELETON_CONNECTIONS:
        if kpt_conf[i] >= threshold and kpt_conf[j] >= threshold:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            color = SKELETON_COLORS.get((i, j), (0, 255, 0))
            cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    for idx in range(len(keypoints)):
        if kpt_conf[idx] >= threshold:
            color = KPT_COLOR_HEAD if idx < 5 else KPT_COLOR_BODY
            center = (int(keypoints[idx][0]), int(keypoints[idx][1]))
            cv2.circle(img, center, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, center, 5, (255, 255, 255), 1, cv2.LINE_AA)
