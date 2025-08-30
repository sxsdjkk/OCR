# PaddleOCR FastAPI 服务

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

## 📁 项目结构

```
.
├── app.py                 # FastAPI 主应用
├── start_server.py        # 启动脚本
├── requirements.txt       # Python 依赖
├── Dockerfile            # Docker 镜像配置
├── docker-compose.yml    # Docker 编排配置
├── .dockerignore         # Docker 构建忽略文件
├── build_and_run.sh      # 构建和运行脚本
├── manage_wheel.sh       # Wheel 文件管理脚本
├── test_*.py             # 测试文件
└── data/                 # 数据目录
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
- **OCR 识别**: `POST /ocr_simple/file`
- **结构化 OCR**: `POST /ocr_structure/file`
- **Base64 图片**: `POST /ocr_simple/base64`

## 📚 文档

- **交互式文档**: http://localhost:8008/docs
- **ReDoc 文档**: http://localhost:8008/redoc

## 🐳 Docker 特性

- 基于 NVIDIA CUDA 11.8 镜像
- 支持 GPU 加速
- 健康检查
- 持久化缓存
- 优化的构建过程

## 📝 注意事项

- 需要 NVIDIA GPU 和 CUDA 支持
- Wheel 文件大小约 1.1GB
- 首次运行会下载 PaddleOCR 模型
- 建议使用 SSD 存储以提高性能
