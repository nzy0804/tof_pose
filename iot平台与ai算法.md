# AI 模型对接确认清单（IoT 平台视角）

> **项目**: 智能视频分析 IoT 平台  
> **版本**: v1  
> **日期**: 2026-04-21  
> **用途**: 明确 AI 模型需要支持的接口、参数和性能指标，以便 IoT 平台正确实现调用逻辑

---

## ✅ IoT 平台已确认的模型能力
> 模型输入一张图片，输出S11-S14、S21-S24四张图片和统计数据。

| 序号 | 确认项 | IoT 平台据此实现 | 状态 |
|------|--------|-----------------|------|
| 1 | **模型架构**：**1 个模型输出 S1-S4 + 统计数据** | 平台只需调用 1 个模型服务 | ✅ |
| 2 | 模型处理输入 | 100×100 PNG 原图 | ✅ |
| 3 | 模型输出格式 | S11-S14、S21-S24: 320×320 PNG，统计: JSON | ✅ |
| 4 | 输出大小 | S11-S14、S21-S24 各 30-100KB | ✅ |



---

## 📋 模型输出参数确认

### S1-S4 图像输出（两帧）

| 参数 | 类型 | S1 灰度深度 | S2 彩色深度 | S3 人体骨骼 | S4 人体轮廓 |
|------|------|------------|------------|------------|------------|
| 输出分辨率 | 固定 | 320×320 | 320×320 | 320×320 | 320×320 |
| 输出格式 | 固定 | PNG | PNG | PNG | PNG |
| 单张大小 | 范围 | 30-100KB | 30-100KB | 30-100KB | 30-100KB |



## 📋 IoT 平台调用模型的接口设计

### 方案 : gRPC 调用

```protobuf
// 模型推理服务
service ModelService {
  // 单帧推理
  rpc Infer(InferRequest) returns (InferResponse);
  
}

message InferRequest {
  string frame_id = 1;
  bytes image_data = 2;        // PNG 二进制
}

message InferResponse {
  string frame_id = 1;
  bytes output_image_S11 = 2;
  bytes output_image_S12 = 3;
  bytes output_image_S13 = 4;
  bytes output_image_S14 = 5;
  bytes output_image_S21 = 6;
  bytes output_image_S22 = 7;
  bytes output_image_S23 = 8;
  bytes output_image_S24 = 9;
  int32 person_count = 10;
  int32 processing_time_ms = 11;
}
```

## 🔄 IoT 平台与 AI 模型交互流程

### 方案 : gRPC 调用

#### 1. 单帧推理流程

```
IoT 平台                              AI 模型服务
  │                                        │
  ├── InferRequest ──────────────────────▶│
  │   frame_id: "frame_001"                │
  │   image_data: <PNG 100x100>            │
  │                                        │
  │◀── InferResponse ──────────────────────│
  │   frame_id: "frame_001"                │
  │   output_image_S11: <PNG>               │
  │   output_image_S12: <PNG>               │
  │   output_image_S13: <PNG>               │
  │   output_image_S14: <PNG>               │
  │   output_image_S21: <PNG>               │
  │   output_image_S22: <PNG>               │
  │   output_image_S23: <PNG>               │
  │   output_image_S24: <PNG>               │
  │   person_count: 2                      │
  │   processing_time_ms: 45               │
  │                                        │
```


