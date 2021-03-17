FROM nvidia/cuda:10.1-base-ubuntu18.04

USER root

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.7-dev \
       python3.7 \
       python3-pip \
       python3-setuptools \
       build-essential \
       libgmp-dev \
       libgmp3-dev \
       libssl-dev \
       libffi-dev \
       git \
       tk-dev \
       pandoc

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

RUN python -m pip install --upgrade pip

COPY requirements.txt /code/requirements.txt
RUN python -m pip install -r /code/requirements.txt
RUN python -m pip install numpy==1.20.*

COPY ./docs/docs_requirements.txt /code/docs_requirements.txt
RUN python -m pip install -r /code/docs_requirements.txt


# Copy code once requirements are installed to speed up docker builds when only the code and not the
# requirements have changed.
COPY . /code

USER ubuntu

# Work around for SNC not installing properly (incomplete setup.py)
ENV PYTHONPATH /code
