# ==========================================
# Base image
# ==========================================

FROM python:3.9-slim


# ==========================================
# Environment variables
# ==========================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics


# ==========================================
# Working directory
# ==========================================

WORKDIR /app


# ==========================================
# System dependencies
# ==========================================

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ==========================================
# Create Ultralytics config directory
# ==========================================

RUN mkdir -p /tmp/ultralytics && \
    chmod 777 /tmp/ultralytics


# ==========================================
# Python dependencies
# ==========================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ==========================================
# Copy project files
# ==========================================

COPY tracker_deepsort.py .
COPY pt_baseline.py .
COPY onnx_baseline.py .
COPY compare_results.py .


# ==========================================
# Create results directory
# ==========================================

RUN mkdir -p /app/results


# ==========================================
# Default command
# ==========================================

CMD ["python", "tracker_deepsort.py"]