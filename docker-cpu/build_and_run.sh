#!/bin/bash

# PaddleOCR API CPU版本 - 构建和运行脚本
# 使用方法: ./build_and_run.sh [build|start|stop|restart|logs|status]

set -e

PROJECT_NAME="paddleocr-app-cpu"
COMPOSE_FILE="docker-compose.yml"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker环境
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    log_success "Docker 环境检查通过"
}

# 构建镜像
build_image() {
    log_info "开始构建 CPU 版本镜像..."
    docker compose -f $COMPOSE_FILE build --no-cache
    log_success "镜像构建完成"
}

# 启动服务
start_service() {
    log_info "启动 OCR API 服务..."
    docker compose -f $COMPOSE_FILE up -d
    log_success "服务启动完成"

    # 等待服务启动
    log_info "等待服务启动..."
    sleep 10

    # 检查服务健康状态
    check_health
}

# 停止服务
stop_service() {
    log_info "停止 OCR API 服务..."
    docker compose -f $COMPOSE_FILE down
    log_success "服务停止完成"
}

# 重启服务
restart_service() {
    log_info "重启 OCR API 服务..."
    stop_service
    sleep 3
    start_service
}

# 查看日志
show_logs() {
    log_info "显示服务日志..."
    docker compose -f $COMPOSE_FILE logs -f
}

# 查看状态
show_status() {
    log_info "服务状态："
    docker compose -f $COMPOSE_FILE ps

    log_info "健康检查："
    check_health
}

# 健康检查
check_health() {
    max_attempts=30
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8008/health > /dev/null 2>&1; then
            log_success "服务健康检查通过 ✓"
            return 0
        fi

        echo -n "."
        sleep 2
        ((attempt++))
    done

    log_error "服务健康检查失败 ✗"
    log_warning "请检查日志： docker compose logs"
    return 1
}

# 测试API
test_api() {
    log_info "测试 API 端点..."

    # 健康检查
    if curl -f -s http://localhost:8008/health > /dev/null 2>&1; then
        log_success "✓ 健康检查通过"
    else
        log_error "✗ 健康检查失败"
        return 1
    fi

    # 检查API文档
    if curl -f -s http://localhost:8008/docs > /dev/null 2>&1; then
        log_success "✓ API文档可访问"
    else
        log_error "✗ API文档不可访问"
    fi

    log_info "API 测试完成"
    log_info "📖 API文档: http://localhost:8008/docs"
    log_info "🔄 ReDoc文档: http://localhost:8008/redoc"
}

# 清理资源
cleanup() {
    log_info "清理 Docker 资源..."
    docker compose -f $COMPOSE_FILE down -v --rmi all 2>/dev/null || true
    docker system prune -f
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    echo "PaddleOCR API CPU版本 - 管理脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [command]"
    echo ""
    echo "可用命令:"
    echo "  build     构建Docker镜像"
    echo "  start     启动服务"
    echo "  stop      停止服务"
    echo "  restart   重启服务"
    echo "  logs      查看日志"
    echo "  status    查看状态"
    echo "  test      测试API"
    echo "  cleanup   清理资源"
    echo "  all       完整流程 (build + start + test)"
    echo "  help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build    # 构建镜像"
    echo "  $0 start    # 启动服务"
    echo "  $0 all      # 一键部署"
}

# 主函数
main() {
    local command=${1:-"help"}

    case $command in
        build)
            check_docker
            build_image
            ;;
        start)
            check_docker
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        test)
            test_api
            ;;
        cleanup)
            cleanup
            ;;
        all)
            check_docker
            build_image
            start_service
            test_api
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
