# MIRA arm — SO-101 MuJoCo simulation & interactive viewer
#
# Lean Python image for the current code stack (mujoco + numpy + pillow).
# No torch/RL deps: add them when the roadmap (RL training, VLA) lands.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System libs for the MuJoCo rendering backends the scripts use:
#   EGL (NVIDIA GPU hosts), osmesa (CPU software), GLFW/X11 (interactive viewer).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    xvfb \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libglfw3 \
    libosmesa6 \
    libglew2.2 \
    libgomp1 \
    libdbus-1-3 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["bash", "/usr/local/bin/entrypoint.sh"]
CMD ["bash"]