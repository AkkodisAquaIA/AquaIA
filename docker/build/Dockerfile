#Verify base image version
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
RUN mkdir -p /root/.cache/torch/hub 
COPY .cache/ /root/.cache/
WORKDIR /workspace
RUN mkdir ./results 
COPY datasets/coco128 ./datasets/coco128
COPY detection/ ./detection
COPY dataloading/ ./dataloading
COPY main.py .
COPY benchmark_train.py .
COPY requirements-vm.txt .
RUN apt-get update && apt-get install -y \
    nano\
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages -r requirements-vm.txt


