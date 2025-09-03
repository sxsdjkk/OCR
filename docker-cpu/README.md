# PaddleOCR API - CPU版本

这是一个基于CPU的PaddleOCR API Docker配置版本，适用于没有GPU的环境。

## 🚀 特性

- ✅ **CPU支持**：无需GPU即可运行OCR识别
- ✅ **轻量级**：基于Ubuntu镜像，无需CUDA依赖
- ✅ **完整功能**：支持所有OCR功能，包括方向矫正
- ✅ **易部署**：一键Docker部署
- ✅ **健康检查**：内置健康检查机制

## 📋 系统要求

- Docker >= 20.0
- Docker Compose >= 2.0
- CPU环境（无需GPU）

## 🛠️ 快速开始

### 1. 构建和启动服务

```bash
# 进入CPU配置目录
cd docker-cpu

# 构建并启动服务
docker compose up -d --build
```

### 2. 验证服务状态

```bash
# 检查容器状态
docker compose ps

# 查看日志
docker compose logs -f
```

### 3. 测试API

```bash
# 健康检查
curl http://localhost:8008/health

# 上传文件测试
curl -X POST "http://localhost:8008/ocr_simple/file" \
     -F "file=@your_image.jpg"
```

## 📁 项目结构

```
docker-cpu/
├── Dockerfile              # CPU版本的Docker镜像配置
├── docker-compose.yml      # CPU版本的服务编排配置
├── requirements.txt        # Python依赖包列表
├── README.md              # 本说明文档
├── app/                   # FastAPI应用代码
│   ├── controllers/
│   ├── services/
│   └── utils/
├── main.py                # 应用入口文件
└── start_server.py       # 服务器启动脚本
```

## ⚙️ 配置说明

### 环境变量

- `PADDLE_PDX_CACHE_HOME=/root/.paddlex` - PaddleX缓存目录
- `CUDA_VISIBLE_DEVICES=""` - 禁用GPU（CPU模式）

### 端口映射

- **8008**: API服务端口

### 数据卷

- `../data:/app/data` - 数据目录映射
- `../models:/root/.paddleocr` - OCR模型目录映射
- `paddle_cache:/root/.paddlex` - PaddleX缓存持久化

## 🔧 自定义配置

### 修改端口

编辑 `docker-compose.yml` 文件中的端口映射：

```yaml
ports:
  - "8080:8008"  # 修改为其他端口
```

### 添加环境变量

在 `docker-compose.yml` 中添加自定义环境变量：

```yaml
environment:
  - PADDLE_PDX_CACHE_HOME=/root/.paddlex
  - CUDA_VISIBLE_DEVICES=""
  - CUSTOM_VAR=value
```

## 📊 性能对比

| 配置版本 | 硬件要求 | 启动时间 | 推理速度 | 内存占用 |
|---------|---------|---------|---------|---------|
| GPU版本 | NVIDIA GPU | ~2分钟 | 快 (~100ms/张) | 中等 |
| CPU版本 | CPU | ~3分钟 | 中 (~500ms/张) | 较低 |

## 🐛 故障排除

### 常见问题

1. **启动失败**
   ```bash
   # 检查Docker状态
   docker system df

   # 清理Docker缓存
   docker system prune -f
   ```

2. **内存不足**
   ```bash
   # 增加Docker内存限制
   # 编辑 /etc/docker/daemon.json
   {
     "default-ulimits": {
       "nofile": {
         "Name": "nofile",
         "Hard": 65536,
         "Soft": 65536
       }
     }
   }
   ```

3. **模型下载失败**
   ```bash
   # 手动创建模型目录
   mkdir -p ../models
   ```

## 🔄 从GPU版本迁移

如果你之前使用GPU版本，想要切换到CPU版本：

1. 停止GPU服务：
   ```bash
   cd ..  # 返回上级目录
   docker compose down
   ```

2. 启动CPU服务：
   ```bash
   cd docker-cpu
   docker compose up -d --build
   ```

## 📝 API文档

启动服务后，访问以下地址查看API文档：

- **Swagger UI**: http://localhost:8008/docs
- **ReDoc**: http://localhost:8008/redoc

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个CPU版本配置。

## 📄 许可证

本项目采用与主项目相同的许可证。
