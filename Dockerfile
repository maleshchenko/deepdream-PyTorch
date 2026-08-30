FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url "${TORCH_INDEX_URL}"

COPY test_pytorch.py pyproject.toml ./

ENV TORCH_HOME=/root/.cache/torch
RUN python -c "from torchvision import models; models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)"

WORKDIR /data
ENTRYPOINT ["python", "/app/test_pytorch.py"]
