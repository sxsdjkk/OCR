# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PADDLE_PDX_CACHE_HOME=/root/.paddlex
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Download paddlepaddle wheel file if it doesn't exist
RUN if [ ! -f "paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl" ]; then \
        echo "Downloading paddlepaddle wheel file..." && \
        wget -O paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl \
        "https://paddle-whl.bj.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl" && \
        echo "Download completed successfully!"; \
    else \
        echo "PaddlePaddle wheel file already exists, skipping download."; \
    fi

# Install paddlepaddle wheel file
RUN pip3 install --no-cache-dir paddlepaddle_gpu-3.0.0-cp310-cp310-manylinux1_x86_64.whl

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies (excluding paddlepaddle-gpu for now)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.paddleocr /root/.paddlex /app/data

# Set permissions
RUN chmod +x app.py start_server.py

# Expose port for web applications
EXPOSE 8008

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

# Default command - can be overridden
CMD ["python3", "start_server.py"]
