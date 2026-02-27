#Verify base image version
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime 

RUN mkdir /downloads
COPY aquaia_archive.tar /downloads/aquaia_archive.tar

RUN mkdir /usr/local/etc/AquaIA

RUN tar -xvf /downloads/aquaia_archive.tar -C /usr/local/etc/AquaIA
# Outils de build nécessaires à hdbscan (compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Outils python pour builder des wheels
RUN pip install --no-cache-dir -U pip setuptools wheel

# Très souvent nécessaire selon les versions (évite des builds cassés)
RUN pip install --no-cache-dir "numpy<2" Cython
RUN pip install --no-cache-dir -r /usr/local/etc/AquaIA/requirements.txt
RUN pip install -r /usr/local/etc/AquaIA/requirements-dev.txt
RUN pip install -r /usr/local/etc/AquaIA/requirements-gpu.txt