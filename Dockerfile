FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ffmpeg ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 7860
CMD ["python3", "-m", "app.main"]
