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

RUN python -m pip install -U pip && pip install poetry

## Install development dependencies into the virtual environment
# There is an issue installing some of the dependencies - switch off the new installer
RUN poetry config experimental.new-installer false
COPY poetry.lock pyproject.toml /tmp/src/snc/
RUN cd /tmp/src/snc && poetry install -n --no-root

# Install demand_planning_service into the virtual environment
COPY ./ /tmp/src/snc/

RUN cd /tmp/src/snc \
    && poetry build -f wheel -n \
    && pip install --no-deps dist/*.whl

RUN pip install -r /tmp/src/snc/docs/docs_requirements.txt
