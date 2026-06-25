# MaixSense TensorRT OSS 服务部署流程

本文档用于把本地 MaixSense 项目源码上传到 AI 服务器，在服务器已有 TensorRT 环境中编译 PyInstaller 可执行文件，然后用 `nohup` 重启 gRPC 服务，最后删除服务器上的临时源码。

重要原则：

- 不使用 `systemd`。
- 不在命令、文档、日志里写密码、AccessKey、Secret。
- 编译只在服务器临时目录 `/data/care/build` 下进行。
- 部署后只删除临时源码包和临时编译目录，不删除运行目录、模型 engine、日志、PID 文件或 OSS env 文件。

## 1. 当前固定路径

本地项目：

```text
E:\Project\Lab Project\MaixSense
```

AI 服务器：

```text
care@222.71.62.147
```

服务器部署目录：

```text
/data/care/care-sense-iot-platform/bin
```

服务器编译工作目录：

```text
/data/care/build
```

服务器 TensorRT 编译 Python：

```text
/data/care/tensorrt-export-venv/bin/python
```

当前服务软链接：

```text
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-current
```

OSS 环境变量文件：

```text
/data/care/care-sense-iot-platform/bin/maixsense-oss.env
```

batch-20 engine：

```text
/data/care/trt-export-lowmem-20260612-b20w1/maixsense-seg-lowmem-b20w1.engine
/data/care/trt-export-lowmem-20260612-b20w1/maixsense-pose-lowmem-b20w1.engine
```

## 2. 本地编译前检查

下面命令在 Windows PowerShell 中执行。

进入项目目录：

```powershell
cd "E:\Project\Lab Project\MaixSense"
```

如果修改过 `ai.proto`，先重新生成 protobuf 文件：

```powershell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ai.proto
```

做 Python 语法检查：

```powershell
python -m py_compile `
  scripts\grpc_server.py `
  src\tof_pose\realtime_service.py `
  src\tof_pose\object_storage.py `
  ai_pb2.py `
  ai_pb2_grpc.py
```

确认当前本地改动：

```powershell
git status --short
git rev-parse HEAD
```

## 3. 打包本地源码

只打包可执行文件需要的源码，不打包 `.git`、虚拟环境、样本数据、模型文件和历史部署包。

在 PowerShell 中执行：

```powershell
$purpose = "your-purpose"
$ts = Get-Date -Format "yyyyMMddHHmmss"
$name = "maixsense-deploy-$purpose-$ts.tar.gz"

tar -czf $name `
  ai.proto `
  ai_pb2.py `
  ai_pb2_grpc.py `
  scripts `
  src `
  maixsense-grpc-server-trt.spec `
  maixsense-grpc-server.spec

Write-Output $name
```

`$purpose` 用简短英文或拼音，例如：

```text
distance-empty-single-person
oss-timeout-test
balanced-binding
```

## 4. 上传源码包到服务器

先确保服务器编译目录存在：

```powershell
ssh care@222.71.62.147 "mkdir -p /data/care/build"
```

上传刚才生成的 tar 包：

```powershell
scp $name care@222.71.62.147:/data/care/build/$name
```

上传完成后，可以在服务器检查：

```powershell
ssh care@222.71.62.147 "ls -lh /data/care/build/$name"
```

## 5. 在服务器解压源码

下面命令在 AI 服务器上执行。可以先登录：

```powershell
ssh care@222.71.62.147
```

登录后执行：

```bash
PURPOSE=your-purpose
TS=$(date +%Y%m%d%H%M%S)
BUILD_ID=maixsense-${PURPOSE}-build-${TS}
BUILD_DIR=/data/care/build/$BUILD_ID
REMOTE_TAR=/data/care/build/maixsense-deploy-${PURPOSE}-<local-ts>.tar.gz

mkdir -p /data/care/build
rm -rf -- "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
tar -xzf "$REMOTE_TAR" -C "$BUILD_DIR"
```

注意把 `<local-ts>` 替换成本地 tar 包文件名里的时间戳。也可以直接设置完整文件名：

```bash
REMOTE_TAR=/data/care/build/maixsense-deploy-your-purpose-20260623123456.tar.gz
```

## 6. 在服务器编译

进入解压目录：

```bash
cd "$BUILD_DIR"
```

推荐后台编译，避免 SSH 连接中断导致你误以为需要重新编译：

```bash
rm -f build.exit build.pid
(
  PYTHONUNBUFFERED=1 /data/care/tensorrt-export-venv/bin/python \
    -m PyInstaller --clean --noconfirm maixsense-grpc-server-trt.spec \
    > pyinstaller-build.log 2>&1
  echo $? > build.exit
) & echo $! > build.pid

cat build.pid
```

轮询编译状态：

```bash
cat "$BUILD_DIR/build.exit" 2>/dev/null || true
PID=$(cat "$BUILD_DIR/build.pid" 2>/dev/null || true)
if [ -n "$PID" ]; then ps -fp "$PID" || true; fi
ls -lh "$BUILD_DIR/dist/maixsense-grpc-server/maixsense-grpc-server" 2>/dev/null || true
tail -n 60 "$BUILD_DIR/pyinstaller-build.log" 2>/dev/null || true
```

编译成功时，`build.exit` 应该是：

```text
0
```

并且存在：

```text
$BUILD_DIR/dist/maixsense-grpc-server/maixsense-grpc-server
```

## 7. 部署新 runtime

编译成功后，把整个 PyInstaller dist 目录移动到部署目录，并更新 current 软链接。

```bash
BIN_DIR=/data/care/care-sense-iot-platform/bin
CURRENT=$BIN_DIR/maixsense-grpc-server-trt-current
RUNTIME=$BIN_DIR/maixsense-grpc-server-trt-runtime-$(date +%Y%m%d)-$PURPOSE

if [ -e "$RUNTIME" ]; then
  echo "runtime already exists: $RUNTIME" >&2
  exit 12
fi

mv "$BUILD_DIR/dist/maixsense-grpc-server" "$RUNTIME"
cp "$BUILD_DIR/pyinstaller-build.log" "$RUNTIME-build.log"
ln -sfn "$RUNTIME" "$CURRENT"

readlink -f "$CURRENT"
```

说明：

- `RUNTIME` 是真正运行的 PyInstaller 目录。
- `CURRENT` 是软链接，启动服务时固定执行它下面的 `maixsense-grpc-server`。
- 部署后删除临时源码不会影响服务运行。

## 8. 停止旧服务

不要用过宽的 `grep | kill`，避免误杀当前 SSH 命令。优先通过 PID 文件和 50052 端口停止。

```bash
BIN_DIR=/data/care/care-sense-iot-platform/bin

kill_pid_if_service() {
  pid="$1"
  if [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ]; then
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    case "$cmdline" in
      *maixsense-grpc-server*'--port 50052'*) kill "$pid" || true ;;
    esac
  fi
}

for pid_file in "$BIN_DIR"/maixsense-grpc-server-trt-*.pid; do
  [ -e "$pid_file" ] || continue
  kill_pid_if_service "$(cat "$pid_file" 2>/dev/null || true)"
done

for pid in $(ss -lntp 2>/dev/null | sed -n 's/.*:50052.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u); do
  kill_pid_if_service "$pid"
done

sleep 5

for pid in $(ss -lntp 2>/dev/null | sed -n 's/.*:50052.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u); do
  kill -9 "$pid" || true
done

sleep 2
ss -lntp | grep ':50052' || true
```

如果最后没有输出 `:50052`，说明旧服务已经停掉。

## 9. 启动 OSS 版本服务

下面是当前 OSS object-key 模式常用启动命令。

```bash
BIN_DIR=/data/care/care-sense-iot-platform/bin
CURRENT=$BIN_DIR/maixsense-grpc-server-trt-current
ENV_FILE=$BIN_DIR/maixsense-oss.env
LOG=$BIN_DIR/maixsense-grpc-server-trt-b20-$PURPOSE.log
PID_FILE=$BIN_DIR/maixsense-grpc-server-trt-b20-$PURPOSE.pid
SEG=/data/care/trt-export-lowmem-20260612-b20w1/maixsense-seg-lowmem-b20w1.engine
POSE=/data/care/trt-export-lowmem-20260612-b20w1/maixsense-pose-lowmem-b20w1.engine

cd "$BIN_DIR"

set -a
. "$ENV_FILE"
set +a

: > "$LOG"
nohup env PYTHONUNBUFFERED=1 "$(printf '\131\117\114\117_AUTOINSTALL')=false" "$CURRENT/maixsense-grpc-server" \
  --host 0.0.0.0 \
  --port 50052 \
  --max-workers 6 \
  --max-msg-mb 160 \
  --device cuda:0 \
  --decode-workers 6 \
  --render-workers 6 \
  --model-instances 6 \
  --warmup-batch-size 20 \
  --device-binding-ttl-sec 0 \
  --output-format jpeg \
  --jpeg-quality 60 \
  --cpu-worker-mode process \
  --cpu-process-start-method fork \
  --model-path "$SEG" \
  --pose-model-path "$POSE" \
  --contour-new-conf 0.25 \
  --pose-gate-kpt-conf 0.35 \
  --pose-kpt-min-points 4 \
  --mask-max-area-ratio 0.90 \
  --oss-workers 4 \
  --oss-download-workers 4 \
  --oss-upload-workers 8 \
  --oss-global-workers 8 \
  --oss-max-pool-connections 128 \
  > "$LOG" 2>&1 & echo $! > "$PID_FILE"
```

不要打印 `maixsense-oss.env` 内容。

## 10. 验证服务

检查 PID：

```bash
cat "$PID_FILE"
PID=$(cat "$PID_FILE" 2>/dev/null || true)
if [ -n "$PID" ]; then ps -fp "$PID"; fi
```

检查端口：

```bash
ss -lntp | grep ':50052'
```

看启动日志：

```bash
tail -n 120 "$LOG"
```

正常应看到类似信息：

```text
Initializing AI model instance
Warmup timing
Starting gRPC server on 0.0.0.0:50052
```

有请求进来后，OSS 模式日志里应出现：

```text
Infer batch timing
Infer request timing
oss_download_ms
oss_upload_ms
```

如果服务启动失败，先看：

```bash
tail -n 200 "$LOG"
```

常见检查项：

- engine 路径是否存在。
- `maixsense-oss.env` 是否存在且权限正确。
- 50052 端口是否被旧进程占用。
- 当前 `CURRENT` 是否指向刚部署的新 runtime。

## 11. 删除服务器上的临时源码

只有在新 runtime 部署、服务启动和端口验证完成后，再删除临时源码。

```bash
rm -rf -- "$BUILD_DIR" "$REMOTE_TAR"
```

再次确认没有遗留临时源码：

```bash
find /data/care/build -maxdepth 1 -name "$BUILD_ID*" -print || true
```

如果没有输出，说明本次临时源码目录和源码包已删干净。

不要删除这些内容：

```text
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-runtime-*
/data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-current
/data/care/trt-export-lowmem-*
/data/care/care-sense-iot-platform/bin/maixsense-oss.env
/data/care/care-sense-iot-platform/bin/*.log
/data/care/care-sense-iot-platform/bin/*.pid
```

## 12. 删除本地临时 tar 包

回到 Windows PowerShell：

```powershell
Remove-Item $name
```

这只删除本地临时打包文件，不会删除本地源码。

## 13. 快速回滚方式

如果新版本启动失败，但旧 runtime 目录还在，可以把 `CURRENT` 软链接切回旧 runtime。

先列出已有 runtime：

```bash
ls -ld /data/care/care-sense-iot-platform/bin/maixsense-grpc-server-trt-runtime-*
```

切回某个旧版本：

```bash
BIN_DIR=/data/care/care-sense-iot-platform/bin
CURRENT=$BIN_DIR/maixsense-grpc-server-trt-current
OLD_RUNTIME=$BIN_DIR/maixsense-grpc-server-trt-runtime-YYYYMMDD-old-purpose

ln -sfn "$OLD_RUNTIME" "$CURRENT"
readlink -f "$CURRENT"
```

然后按第 8 节停止旧进程，再按第 9 节启动服务。

## 14. 最小流程清单

每次部署可以按这个顺序检查：

```text
1. 本地修改代码
2. 如果 ai.proto 改了，重新生成 ai_pb2.py 和 ai_pb2_grpc.py
3. 本地 py_compile 检查
4. 本地 tar 打包源码
5. scp 上传到 /data/care/build
6. 服务器解压到新的 BUILD_DIR
7. 服务器 TensorRT 环境 PyInstaller 编译
8. 移动 dist 到新的 runtime 目录
9. 更新 maixsense-grpc-server-trt-current 软链接
10. 停止旧 50052 服务
11. nohup 启动新服务
12. 检查 PID、端口、日志
13. 删除服务器临时源码包和 BUILD_DIR
14. 删除本地临时 tar 包
```

