#!/bin/bash

# PaddlePaddle Wheel 文件管理脚本

WHEEL_FILE="paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl"
DOWNLOAD_URL="https://paddle-whl.bj.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl"

echo "🔧 PaddlePaddle Wheel 文件管理工具"
echo "=================================="

if [ -f "$WHEEL_FILE" ]; then
    echo "✅ Wheel 文件已存在: $WHEEL_FILE"
    echo "📊 文件大小: $(du -h "$WHEEL_FILE" | cut -f1)"
    echo ""
    echo "选择操作:"
    echo "1) 保留现有文件"
    echo "2) 删除现有文件并重新下载"
    echo "3) 验证文件完整性"
    read -p "请输入选择 (1-3): " choice
    
    case $choice in
        1)
            echo "✅ 保留现有文件"
            ;;
        2)
            echo "🗑️  删除现有文件..."
            rm "$WHEEL_FILE"
            echo "📥 开始下载..."
            wget -O "$WHEEL_FILE" "$DOWNLOAD_URL"
            if [ $? -eq 0 ]; then
                echo "✅ 下载完成！"
            else
                echo "❌ 下载失败！"
                exit 1
            fi
            ;;
        3)
            echo "🔍 验证文件完整性..."
            if [ -s "$WHEEL_FILE" ]; then
                echo "✅ 文件存在且非空"
                echo "📊 文件大小: $(du -h "$WHEEL_FILE" | cut -f1)"
            else
                echo "❌ 文件可能损坏或为空"
            fi
            ;;
        *)
            echo "❌ 无效选择"
            exit 1
            ;;
    esac
else
    echo "❌ Wheel 文件不存在: $WHEEL_FILE"
    echo ""
    echo "选择操作:"
    echo "1) 下载 wheel 文件"
    echo "2) 跳过下载（Docker 构建时会自动下载）"
    read -p "请输入选择 (1-2): " choice
    
    case $choice in
        1)
            echo "📥 开始下载..."
            wget -O "$WHEEL_FILE" "$DOWNLOAD_URL"
            if [ $? -eq 0 ]; then
                echo "✅ 下载完成！"
                echo "📊 文件大小: $(du -h "$WHEEL_FILE" | cut -f1)"
            else
                echo "❌ 下载失败！"
                exit 1
            fi
            ;;
        2)
            echo "⏭️  跳过下载，Docker 构建时会自动下载"
            ;;
        *)
            echo "❌ 无效选择"
            exit 1
            ;;
    esac
fi

echo ""
echo "💡 提示:"
echo "- 如果选择保留或下载文件，Docker 构建会更快"
echo "- 如果不下载，Docker 构建时会自动下载（但会延长构建时间）"
echo "- 文件大小约为 1.1GB，请确保有足够的磁盘空间"
