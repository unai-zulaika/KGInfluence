FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --allow-unauthenticated \
    build-essential \
    python3-dev \
    python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install -U torch numpy
