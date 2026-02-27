#Verify base image version
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime 

RUN mkdir /downloads
COPY aquaia_archive.tar /downloads/aquaia_archive.tar

RUN mkdir /usr/local/etc/AquaIA

RUN tar -xvf /root/downloads/aquaia_archive.tar -C /usr/local/etc/AquaIA
RUN pip install -r /usr/local/etc/AquaIA/requirements.txt
RUN pip install -r /usr/local/etc/AquaIA/requirements-dev.txt
RUN pip install -r /usr/local/etc/AquaIA/requirements-gpu.txt