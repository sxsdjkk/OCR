#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Startup script for FastAPI PaddleOCR service
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI application"""
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    
    print("Starting FastAPI PaddleOCR Service...")
    print("=" * 50)
    print("API will be available at: http://localhost:8000")
    print("Interactive API docs: http://localhost:8000/docs")
    print("Redoc documentation: http://localhost:8000/redoc")
    print("=" * 50)
    
    try:
        # Run the FastAPI app with uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting the server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()