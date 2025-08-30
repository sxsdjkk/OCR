#!/bin/bash

# PaddleOCR Docker 构建和运行脚本

set -e

echo "🚀 开始构建 PaddleOCR Docker 镜像..."

# 检查 wheel 文件
WHEEL_FILE="paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl"
if [ ! -f "$WHEEL_FILE" ]; then
    echo "⚠️  警告: $WHEEL_FILE 不存在"
    echo "📥 Docker 构建时会自动下载，但会延长构建时间"
    echo "💡 建议先运行 ./manage_wheel.sh 下载文件"
    read -p "是否继续构建？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 构建已取消"
        exit 1
    fi
else
    echo "✅ 找到 wheel 文件: $WHEEL_FILE"
    echo "📊 文件大小: $(du -h "$WHEEL_FILE" | cut -f1)"
fi

# 构建镜像
docker build -t paddleocr-app:latest .

echo "✅ 镜像构建完成！"

# 询问是否立即运行
read -p "是否立即运行容器？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🐳 启动 PaddleOCR 容器..."
    
    # 创建必要的目录
    mkdir -p data models
    
    # 运行容器
    docker-compose up -d
    
    echo "✅ 容器启动成功！"
    echo "📱 API 地址: http://localhost:8008"
    echo "📚 文档地址: http://localhost:8008/docs"
    echo "🔍 健康检查: http://localhost:8008/health"
    echo ""
    echo "查看日志: docker-compose logs -f"
    echo "停止服务: docker-compose down"
else
    echo "💡 使用以下命令运行容器："
    echo "   docker-compose up -d"
    echo "   docker run -d -p 8008:8008 --gpus all paddleocr-app:latest"
fi
