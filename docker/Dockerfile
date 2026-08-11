# Animal AI 1st place solution (NeurIPS 2019) reproduction environment.
#
# The code is pinned to TensorFlow 1.14 / Python 3.6, which needs CUDA 10.0 +
# cuDNN 7. NVIDIA removed the `nvidia/cuda:10.0-*` images from Docker Hub, so
# the CUDA runtime libraries are installed from NVIDIA's .deb archive instead.
FROM ubuntu:18.04

ARG CUDA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
ARG ML_REPO=https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64
ARG CUDA_PKG_VERSION=10.0.130-1
ARG CUDNN_PKG_VERSION=7.6.5.32-1+cuda10.0
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Python 3.6 plus the X11/GL libraries the Unity 2018.3 player dlopens at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        unzip \
        git \
        python3.6 \
        python3.6-dev \
        python3-distutils \
        xvfb \
        x11-utils \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        libglu1-mesa \
        libx11-6 \
        libxcursor1 \
        libxrandr2 \
        libxinerama1 \
        libxi6 \
        libxxf86vm1 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libasound2 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        libxv1 \
        libxtst6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Xvfb only offers a software GL 3.1 compat profile, which the Unity 2018.3
# player rejects ("Unable to find a supported OpenGL core profile"). VirtualGL
# renders through the NVIDIA driver over EGL and blits the result into Xvfb.
ARG VIRTUALGL_VERSION=3.1
RUN curl -fsSLO "https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" \
    && apt-get update \
    && apt-get install -y --no-install-recommends "./virtualgl_${VIRTUALGL_VERSION}_amd64.deb" \
    && rm -f "virtualgl_${VIRTUALGL_VERSION}_amd64.deb" \
    && rm -rf /var/lib/apt/lists/*

# libglvnd needs an ICD entry to hand EGL over to the NVIDIA driver. The
# container toolkit usually mounts this file; write it so the image also works
# when it does not.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : {\n    "library_path" : "libEGL_nvidia.so.0"\n  }\n}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# CUDA 10.0 runtime libraries + cuDNN 7 (only what TensorFlow 1.14 loads).
RUN cd /tmp \
    && for pkg in cuda-license-10-0 cuda-cudart-10-0 cuda-cublas-10-0 cuda-cufft-10-0 \
                  cuda-curand-10-0 cuda-cusolver-10-0 cuda-cusparse-10-0; do \
           curl -fsSLO "${CUDA_REPO}/${pkg}_${CUDA_PKG_VERSION}_amd64.deb"; \
       done \
    && curl -fsSLO "${ML_REPO}/libcudnn7_${CUDNN_PKG_VERSION}_amd64.deb" \
    && dpkg -i /tmp/*.deb \
    && rm -f /tmp/*.deb

ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# pip 21.3.1 is the last release that supports Python 3.6.
RUN curl -fsSL https://bootstrap.pypa.io/pip/3.6/get-pip.py -o /tmp/get-pip.py \
    && python3.6 /tmp/get-pip.py "pip==21.3.1" "setuptools==57.5.0" "wheel==0.37.1" \
    && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.6 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.6 /usr/local/bin/python3

COPY requirements.txt /tmp/requirements.txt
RUN python3.6 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python3.6 -m pip install --no-cache-dir \
        "requests==2.27.1" \
        "psutil==5.9.8" \
        "setproctitle==1.2.3" \
        "notebook<7" \
        "ipywidgets<8" \
    && rm /tmp/requirements.txt

# Run as the host user so files written to the bind-mounted repo stay editable.
RUN groupadd -g ${GROUP_ID} animal \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash animal

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV DISPLAY=:99 \
    XVFB_SCREEN=1280x1024x24 \
    PYTHONPATH=/workspace \
    PATH=/opt/VirtualGL/bin:${PATH} \
    VGL_DEVICE=egl0

USER animal
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
