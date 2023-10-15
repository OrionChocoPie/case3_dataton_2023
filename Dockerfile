FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update -y

WORKDIR /workdir

COPY ./ /workdir

RUN pip install --no-cache-dir --upgrade -r requirements.txt