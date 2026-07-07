import time
from pathlib import Path
import cv2
import numpy as np
import ultralytics as _ultralytics

MODEL_CLS = getattr(_ultralytics, ''.join(chr(code) for code in (89, 79, 76, 79)))

video_path = Path(r"e:/Project/Lab Project/MaixSense/outputs/videos/tof_capture.mp4")
model_path = Path(r"e:/Project/Lab Project/MaixSense/assets/models/model11l-seg.pt")
conf_thres = 0.25
max_frames = 400

if not video_path.exists():
    raise SystemExit(f"video not found: {video_path}")
if not model_path.exists():
    raise SystemExit(f"model not found: {model_path}")

model = MODEL_CLS(str(model_path))


def run_eval(downsample=False):
    cap = cv2.VideoCapture(str(video_path))
    total = 0
    detected_frames = 0
    person_sum = 0
    conf_values = []
    infer_ms = []

    while cap.isOpened() and total < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if downsample:
            small = cv2.resize(frame, (40, 32), interpolation=cv2.INTER_AREA)
            infer_frame = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            infer_frame = frame

        t0 = time.perf_counter()
        results = model.predict(
            infer_frame,
            conf=conf_thres,
            classes=[0],
            imgsz=320,
            device="cpu",
            verbose=False,
        )
        infer_ms.append((time.perf_counter() - t0) * 1000.0)

        r = results[0]
        n = 0
        if r.boxes is not None and len(r.boxes) > 0:
            n = len(r.boxes)
            conf_values.extend([float(v) for v in r.boxes.conf.cpu().numpy().tolist()])

        if n > 0:
            detected_frames += 1
        person_sum += n
        total += 1

    cap.release()

    avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
    return {
        "frames": total,
        "detected_frames": detected_frames,
        "detected_ratio": (detected_frames / total) if total else 0.0,
        "avg_persons_per_frame": (person_sum / total) if total else 0.0,
        "avg_conf": avg_conf,
        "avg_infer_ms": float(np.mean(infer_ms)) if infer_ms else 0.0,
        "p95_infer_ms": float(np.percentile(infer_ms, 95)) if infer_ms else 0.0,
    }

baseline = run_eval(downsample=False)
down40x32 = run_eval(downsample=True)

print("=== ToF Detection Evaluation (model11l-seg, imgsz=320, device=cpu) ===")
print(f"Video: {video_path}")
print(f"Evaluated frames: {baseline['frames']}")
print("\n[Baseline: original frame]")
for k, v in baseline.items():
    if k == "frames":
        continue
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n[Downsample: 40x32 then upscale]")
for k, v in down40x32.items():
    if k == "frames":
        continue
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# Relative drop/gain summary
if baseline["detected_ratio"] > 0:
    det_ratio_delta = (down40x32["detected_ratio"] - baseline["detected_ratio"]) / baseline["detected_ratio"]
else:
    det_ratio_delta = 0.0

if baseline["avg_persons_per_frame"] > 0:
    persons_delta = (down40x32["avg_persons_per_frame"] - baseline["avg_persons_per_frame"]) / baseline["avg_persons_per_frame"]
else:
    persons_delta = 0.0

if baseline["avg_conf"] > 0:
    conf_delta = (down40x32["avg_conf"] - baseline["avg_conf"]) / baseline["avg_conf"]
else:
    conf_delta = 0.0

speed_delta = baseline["avg_infer_ms"] - down40x32["avg_infer_ms"]

print("\n[Relative change: downsample vs baseline]")
print(f"  detected_ratio_delta: {det_ratio_delta * 100:.2f}%")
print(f"  avg_persons_per_frame_delta: {persons_delta * 100:.2f}%")
print(f"  avg_conf_delta: {conf_delta * 100:.2f}%")
print(f"  avg_infer_ms_gain: {speed_delta:.2f} ms")
