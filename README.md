# tof_pose

`tof_pose` 项目当前围绕 3 个可直接运行的流程组织：

- `scripts/tof_pose.py`：从 MaixSense 串口数据流进行实时姿态推理，并结合深度信息估计每个人到深度相机的距离
- `scripts/tof_record.py`：采集并保存 ToF 伪彩视频
- `scripts/tof_video_inference.py`：对已录制视频执行离线姿态推理
- `scripts/prepare_dataset.py`：从视频抽帧，生成适合 `YOLO pose` 训练的数据集目录
- `scripts/convert_coco_to_yolo_pose.py`：将 CVAT 导出的 COCO Keypoints 标注转换为 YOLO pose 标签

## 功能说明

### 1. 实时姿态推理与距离估计

运行 `scripts/tof_pose.py` 后，程序会：

- 从 MaixSense 串口读取深度数据
- 将深度图转换为伪彩图，送入 YOLO 姿态模型
- 绘制人体骨架关键点
- 根据人体检测框、躯干关键点和深度分布，估计每个人相对深度相机的距离
- 在画面中显示 `P1 Dist~...` 这一类距离标签

画面中人体周围的白色轮廓线表示：

- 当前用于距离估计的人体主体区域轮廓
- 这条轮廓来自深度分割结果，用来辅助观察距离估计是否覆盖到了正确的人体区域

### 2. 视频录制

`scripts/tof_record.py` 会把实时深度图保存为伪彩视频，便于后续离线分析和可视化。

### 3. 离线视频姿态推理

`scripts/tof_video_inference.py` 会对录制好的伪彩视频进行姿态推理，并输出带骨架结果的视频。

注意：

- 当前离线推理流程使用的是伪彩视频，不包含原始深度值
- 因此离线视频推理目前不支持真实距离计算
- 距离估计功能目前只在实时 ToF 数据流中生效

### 4. 数据集整理

`scripts/prepare_dataset.py` 可用于把录制视频抽帧成训练图像，并自动生成标准目录结构：

- `dataset/images/train|val|test`
- `dataset/labels/train|val|test`
- `dataset/meta/manifest.csv`
- `dataset/dataset.yaml`

详细规范与标注检查清单见 [dataset/README.md](dataset/README.md)。

### 5. 标注结果转换

如果你在 CVAT 中导出的是 `COCO Keypoints 1.0`，可以用下面的命令转换成 `YOLO pose` 标签：

```powershell
python scripts/convert_coco_to_yolo_pose.py --input-dir "task_1_dataset_2026_03_31_11_47_13_coco keypoints 1.0" --dataset-root dataset --split train --copy-images --overwrite
```

### 6. 模型训练

将标注结果整理到 `dataset/` 后，可以直接使用本地 pose 预训练模型继续训练：

```powershell
yolo pose train data=dataset/dataset.yaml model=assets/models/yolo11n-pose.pt epochs=30 imgsz=320 batch=8 device=0 project=runs/tof_pose name=pose_debug_v1
```

说明：

- `data=dataset/dataset.yaml`：数据集配置文件
- `model=assets/models/yolo11n-pose.pt`：本地初始 pose 模型
- `imgsz=320`：与当前项目推理分辨率保持一致
- `batch=8`：适合中小显存环境的起步设置
- `device=0`：使用第 1 张 GPU

如果需要更长时间正式训练，可在确认数据集无误后增大 `epochs`。

### 7. 部署新模型

训练完成后，推荐将导出的最佳权重复制到 `assets/models/` 中统一管理，例如：

```text
assets/models/tof_pose_best.pt
```

当前项目默认已经切换到：

```text
assets/models/tof_pose_best.pt
```

默认模型路径定义在：

- `src/tof_pose/paths.py`

因此后续直接运行下面的命令时，会自动使用新训练得到的模型：

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_video_inference.py
```

## 目录结构

- `src/tof_pose/`：可复用的核心业务代码
- `assets/models/`：YOLO 姿态模型权重
- `outputs/videos/`：录制视频与推理结果视频
- `scripts/`：项目运行入口脚本
- `dataset/`：训练数据目录模板与数据集说明
- `runs/`：训练过程输出目录

## 运行示例

```powershell
python scripts/tof_pose.py COM8
python scripts/tof_record.py COM8
python scripts/tof_video_inference.py
python scripts/prepare_dataset.py --input outputs/videos --dataset-root dataset --sample-every 5 --touch-labels
python scripts/convert_coco_to_yolo_pose.py --input-dir "task_1_dataset_2026_03_31_11_47_13_coco keypoints 1.0" --dataset-root dataset --split train --copy-images --overwrite
yolo pose train data=dataset/dataset.yaml model=assets/models/yolo11n-pose.pt epochs=30 imgsz=320 batch=8 device=0 project=runs/tof_pose name=pose_debug_v1
```
