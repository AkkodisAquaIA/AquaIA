#Verify base image version
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime 

RUN mkdir /root/downloads
COPY aquaia_archive.tar /root/downloads/aquaia_archive.tar

RUN tar -xvf /root/downloads/aquaia_archive.tar -C /usr/local/etc/
RUN pip install -r /usr/local/etc/AquaIA/requirements.txt