FROM python:3.10-slim-bookworm

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libosmesa6 libglfw3 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    'torch @ https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl'
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install \
    'numpy==1.26.4' 'mujoco==3.1.6' 'robosuite==1.4.1' 'bddl==1.0.1' \
    'PyYAML==6.0.2' 'imageio==2.37.0' 'tqdm==4.67.1' 'tyro==0.9.19' \
    'websockets==14.2' 'msgpack==1.1.0' 'dm-tree==0.1.9' 'pyarrow==19.0.1'

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install \
    'matplotlib==3.8.4' 'future==0.18.2' 'easydict==1.9' 'gym==0.25.2'

ENV PYTHONPATH=/app:/app/third_party/libero:/app/packages/openpi-client/src \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=/tmp/libero \
    PYTHONDONTWRITEBYTECODE=1
RUN mkdir -p /tmp/libero && printf '%s\n' \
    'benchmark_root: /app/third_party/libero/libero/libero' \
    'bddl_files: /app/third_party/libero/libero/libero/bddl_files' \
    'init_states: /app/third_party/libero/libero/libero/init_files' \
    'datasets: /app/third_party/libero/datasets' \
    'assets: /app/third_party/libero/libero/libero/assets' > /tmp/libero/config.yaml
WORKDIR /app
