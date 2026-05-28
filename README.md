# tof_pose

[English](./README.md) | [中文](./README-ZH.md)

Real-time human pose estimation for MaixSense ToF depth streams, with person distance estimation, pseudo-color video recording, offline inference, and a lightweight training workflow built around YOLO pose.

## Features

- Run real-time pose inference from MaixSense serial depth data
- Estimate each person's distance to the camera from depth cues and person contours
- Record ToF pseudo-color videos for later review and annotation
- Run offline pose inference on recorded videos
- Prepare datasets, convert CVAT annotations, and fine-tune local YOLO pose models

## Quick Start

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_record.py COM8
python scripts/tof_video_inference.py
```

## gRPC Server (Recommended Commands)

Defaults: port `50052`, max message size `50MB`.

### 1) Default mode (keeps cross-call cache/tracking)

Best for continuous streams (serial/video).

```powershell
python scripts/grpc_server.py --host 0.0.0.0 --port 50052 --max-workers 1
```

### 2) Stateless mode (treat every call as the first frame)

Best for single images or sparse frames.

```powershell
python scripts/grpc_server.py --host 0.0.0.0 --port 50052 --max-workers 1 --stateless
```

### 3) ToF pose-only (no segmentation; use your trained pose weights)

If the segmentation model does not detect people on ToF pseudo-color inputs, use pose-only to derive `person_count` from pose keypoints and render skeleton output.

```powershell
python scripts/grpc_server.py --host 0.0.0.0 --port 50052 --max-workers 1 --stateless --pose-only --pose-model-path assets/models/tof_pose_best.pt --pose-conf 0.15
```

Optional knobs for pose-only counting:

- `--pose-kpt-conf` (default 0.20)
- `--pose-kpt-min-points` (default 4)

### 4) Single-image client smoke test

```powershell
python scripts/grpc_client_test.py <input.png> <frame_id> --device-id device_0000 --host 127.0.0.1 --port 50052
```

## Workflow

### Real-Time Pose Inference

`scripts/tof_pose.py` reads MaixSense serial depth frames, converts them into pseudo-color images, runs YOLO pose inference, draws human skeletons, and overlays per-person distance estimates such as `P1 Dist~...`.

The white contour around a person visualizes the body region currently used for distance estimation.

### Video Recording

`scripts/tof_record.py` records the live depth stream as pseudo-color video for later labeling, visualization, and offline inference.

### Offline Video Inference

`scripts/tof_video_inference.py` runs pose inference on recorded pseudo-color videos and writes a skeleton-rendered output video.

Notes:

- The offline pipeline uses pseudo-color video rather than raw depth values.
- Offline inference does not currently support reliable distance estimation.
- Distance estimation is currently available only in the live ToF workflow.

## Dataset Preparation

Extract frames and generate a standard dataset layout:

```powershell
python scripts/prepare_dataset.py --input outputs/videos --dataset-root dataset --sample-every 5 --touch-labels
```

This creates:

- `dataset/images/train|val|test`
- `dataset/labels/train|val|test`
- `dataset/meta/manifest.csv`
- `dataset/dataset.yaml`

See [dataset/README.md](./dataset/README.md) for dataset conventions and the annotation checklist.

## Annotation Conversion

If you export labels from CVAT in `COCO Keypoints 1.0` format, convert them to `YOLO pose` labels with:

```powershell
python scripts/convert_coco_to_yolo_pose.py --input-dir "task_1_dataset_2026_03_31_11_47_13_coco keypoints 1.0" --dataset-root dataset --split train --copy-images --overwrite
```

## Training

Train a pose model from the prepared dataset with:

```powershell
yolo pose train data=dataset/dataset.yaml model=assets/models/yolo11n-pose.pt epochs=30 imgsz=320 batch=8 device=0 project=runs/tof_pose name=pose_debug_v1
```

Parameter notes:

- `data=dataset/dataset.yaml`: dataset config file
- `model=assets/models/yolo11n-pose.pt`: local starting pose model
- `imgsz=320`: matches the current inference resolution
- `batch=8`: a safe starting point for mid-range GPUs
- `device=0`: use the first GPU

## Model Deployment

After training, the recommended deployment target is:

```text
assets/models/tof_pose_best.pt
```

The project is currently configured to use this file by default. The default model path is defined in [src/tof_pose/paths.py](./src/tof_pose/paths.py).

## Project Structure

- `src/tof_pose/`: core application code
- `assets/models/`: YOLO pose model weights
- `outputs/videos/`: recorded videos and inference outputs
- `scripts/`: runnable project entry points
- `dataset/`: dataset template and documentation
- `runs/`: training outputs

## Common Commands

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_record.py COM8
python scripts/tof_video_inference.py
python scripts/prepare_dataset.py --input outputs/videos --dataset-root dataset --sample-every 5 --touch-labels
python scripts/convert_coco_to_yolo_pose.py --input-dir "task_1_dataset_2026_03_31_11_47_13_coco keypoints 1.0" --dataset-root dataset --split train --copy-images --overwrite
yolo pose train data=dataset/dataset.yaml model=assets/models/yolo11n-pose.pt epochs=30 imgsz=320 batch=8 device=0 project=runs/tof_pose name=pose_debug_v1
```
