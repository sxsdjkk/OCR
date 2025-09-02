# OCR Service (FastAPI + PaddleOCR)

基于 PaddleOCR 的 FastAPI 服务，提供 OCR 文字识别功能。

## 🚀 快速开始

### 方法1：使用便捷脚本（推荐）

```bash
# 管理 wheel 文件
./manage_wheel.sh

# 构建和运行
./build_and_run.sh
```

### 方法2：手动构建和运行

```bash
# 构建镜像
docker build -t paddleocr-app:latest .

# 使用 docker-compose 运行
docker-compose up -d

# 或直接使用 docker 运行
docker run -d -p 8008:8008 --gpus all paddleocr-app:latest
```

### 方法3：本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python start_server.py
```

## 📁 目录结构（MVC）

```
app/
  __init__.py           # FastAPI app，挂载路由与日志配置
  controllers/
    ocr_controller.py   # 路由与请求处理
  services/
    ocr_service.py      # 业务逻辑（一次 OCR → 估角 → 可选旋转 → 同步 polys）
  utils/
    image_utils.py      # base64 与图像编解码
    geom_utils.py       # 多边形与旋转工具
    response_utils.py   # JSON 可序列化工具
main.py                 # 本地调试入口（可选）
start_server.py         # 生产启动入口（使用 "app:app"）
```

## 🔧 Wheel 文件管理

PaddlePaddle GPU 版本通过 wheel 文件安装，支持两种方式：

### 自动下载（Docker 构建时）
- 如果本地没有 wheel 文件，Docker 构建时会自动下载
- 下载地址：https://paddle-whl.bj.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl

### 手动管理
```bash
./manage_wheel.sh
```

## 🌐 API 接口

- **健康检查**: `GET /health`
- **OCR 识别（文件上传）**: `POST /ocr_simple/file`
  - Query: `directionCorrection`（bool，默认 false），`needImg`（bool，默认 false）
- **Base64 图片识别**: `POST /ocr_simple/base64`
  - Query: `directionCorrection`（bool），`needImg`（bool）
- **结构化 OCR（文件上传）**: `POST /ocr_structure/file`

## 📚 文档

- **交互式文档**: http://localhost:8008/docs
- **ReDoc 文档**: http://localhost:8008/redoc

说明：容器内端口为 8008，对外请按需映射（示例 `-p 8010:8008`）。

## 🐳 Docker 特性

- 基于 NVIDIA CUDA 11.8 镜像
- 支持 GPU 加速
- 健康检查
- 持久化缓存
- 优化的构建过程

## 📝 说明与建议

- 需要 NVIDIA GPU 和 CUDA 支持
- Wheel 文件大小约 1.1GB
- 首次运行会下载 PaddleOCR 模型
- 建议使用 SSD 存储以提高性能
- `/ocr_simple/file` 较 `/ocr_simple/base64` 传输更高效（base64 体积膨胀 ~33%）
- 当 `needImg=false` 时，响应中 `ImageBase64` 不返回；但 `Angle/Height/Width` 始终返回
- 当 `directionCorrection=true` 时，服务进行方向矫正，并同步旋转返回的 polygons
