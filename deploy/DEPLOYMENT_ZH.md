# MaixSense AI gRPC 服务器部署指南

## 快速部署（Linux 云服务器）

### 1. 在本地生成可执行文件

首先，在 Windows 本地机器上生成可执行文件：

```bash
.\build_executable.bat
```

构建完成后，在 `dist/` 目录下会生成 `maixsense-grpc-server` 可执行文件。

### 2. 上传到服务器

```bash
# 假设服务器地址为 user@server.com，项目目录为 /opt/maixsense
scp dist/maixsense-grpc-server user@server.com:/opt/maixsense/
scp deploy/maixsense-grpc.service user@server.com:/home/user/
```

### 3. 服务器上配置 systemd

登录服务器：

```bash
ssh user@server.com
```

创建系统用户和目录（如果还没有）：

```bash
sudo useradd -m -s /bin/bash maixsense
sudo mkdir -p /opt/maixsense
sudo chown maixsense:maixsense /opt/maixsense
```

设置可执行文件权限：

```bash
sudo chmod +x /opt/maixsense/maixsense-grpc-server
```

复制 systemd service 文件：

```bash
sudo cp ~/maixsense-grpc.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 4. 启动服务

```bash
# 启动服务
sudo systemctl start maixsense-grpc

# 设置开机自启
sudo systemctl enable maixsense-grpc

# 查看服务状态
sudo systemctl status maixsense-grpc

# 查看日志
sudo journalctl -u maixsense-grpc -f
```

### 5. 验证服务正常运行

从本地或其他机器调用 API：

```bash
python scripts/grpc_client_test.py inputs/example.png frame_000001 --host server.com --port 50052
```

或用 curl 检查端口监听状态：

```bash
nc -zv server.com 50052
```

## 配置说明

### service 文件参数

- `User=maixsense`：服务运行的用户
- `WorkingDirectory=/opt/maixsense`：工作目录
- `ExecStart`：启动命令（自动加上 `--host 0.0.0.0 --port 50052`）
- `Restart=always`：服务意外退出时自动重启
- `RestartSec=10`：重启前等待 10 秒
- `CPUQuota=80%`：限制 CPU 使用率为 80%
- `MemoryLimit=4G`：限制内存使用为 4GB

### 自定义配置

如果需要调整配置，编辑 `/etc/systemd/system/maixsense-grpc.service`：

```bash
sudo systemctl edit maixsense-grpc
```

常见修改：

1. **改变监听端口**：
   ```
   ExecStart=/opt/maixsense/maixsense-grpc-server --host 0.0.0.0 --port 50053
   ```

2. **增加最大并发数**：
   ```
   ExecStart=/opt/maixsense/maixsense-grpc-server --host 0.0.0.0 --port 50052 --max-workers 8
   ```

3. **增加消息大小上限**：
   ```
   ExecStart=/opt/maixsense/maixsense-grpc-server --host 0.0.0.0 --port 50052 --max-msg-mb 100
   ```

修改后重新加载和重启服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart maixsense-grpc
```

## 常见问题

### 如何停止服务？

```bash
sudo systemctl stop maixsense-grpc
```

### 如何重启服务？

```bash
sudo systemctl restart maixsense-grpc
```

### 如何查看详细日志？

```bash
sudo journalctl -u maixsense-grpc -n 100  # 查看最近 100 条日志
sudo journalctl -u maixsense-grpc -f      # 实时跟踪日志
```

### 如何卸载服务？

```bash
sudo systemctl stop maixsense-grpc
sudo systemctl disable maixsense-grpc
sudo rm /etc/systemd/system/maixsense-grpc.service
sudo systemctl daemon-reload
```

## 监控和维护

### 定期检查服务状态

```bash
systemctl is-active maixsense-grpc  # 查看是否运行
systemctl is-enabled maixsense-grpc  # 查看是否开机自启
```

### 监听端口占用

```bash
sudo netstat -tlnp | grep 50052
sudo lsof -i :50052
```

### 检查资源占用

```bash
ps aux | grep maixsense-grpc
```

## 网络配置

如果服务器在防火墙后，需要开放端口 50052：

```bash
# UFW（Debian/Ubuntu）
sudo ufw allow 50052/tcp

# iptables（CentOS/RHEL）
sudo firewall-cmd --permanent --add-port=50052/tcp
sudo firewall-cmd --reload
```

## 反向代理配置（可选）

如果需要通过反向代理访问，以 Nginx 为例：

```nginx
upstream maixsense_grpc {
    server localhost:50052;
}

server {
    listen 80;
    server_name model.example.com;

    # gRPC 需要 HTTP/2 支持
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        grpc_pass grpc://maixsense_grpc;
    }
}
```

## 总结

部署步骤：
1. 本地 `build_executable.bat` 生成可执行文件
2. 上传到服务器 `/opt/maixsense/`
3. 复制 `maixsense-grpc.service` 到 `/etc/systemd/system/`
4. `sudo systemctl start maixsense-grpc`
5. 验证：`sudo systemctl status maixsense-grpc`

就这样。代码和模型权重都不用传，只传一个可执行文件即可。
