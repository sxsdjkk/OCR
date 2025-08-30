#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Startup script for FastAPI PaddleOCR service
"""

import uvicorn
import os
import sys
import logging

def main():
    """Start the FastAPI application"""
    
    # 配置应用日志 - 不覆盖 uvicorn 的日志配置
    app_logger = logging.getLogger("paddleocr_app")
    app_logger.setLevel(logging.INFO)
    
    # 如果还没有处理器，添加一个
    if not app_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [APP] %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        app_logger.addHandler(console_handler)
    
    # 设置传播，让日志也显示在根日志器中
    app_logger.propagate = True
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    # 设置环境变量
    os.environ['PADDLE_PDX_CACHE_HOME'] = '/root/.paddlex'
    
    print("Starting FastAPI PaddleOCR Service...")
    print("=" * 50)
    print("API will be available at: http://localhost:8008")
    print("Interactive API docs: http://localhost:8008/docs")
    print("Redoc documentation: http://localhost:8008/redoc")
    print("=" * 50)
    
    # 测试日志配置
    app_logger.info("应用日志系统已配置完成")
    app_logger.info("准备启动 uvicorn 服务器")
    
    try:
        # Run the FastAPI app with uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8008,
            reload=True,
            log_level="info",  # 保留 uvicorn 的日志
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting the server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()