FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/workspace/.cache/torch

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install CPU wheels for torch/torchaudio (matching versions)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu torchaudio==2.0.2+cpu

# Pin NumPy < 2 to avoid ABI/runtime issues with older wheels
RUN pip install --no-cache-dir "numpy<2"

# Install Demucs deps (will not upgrade torch/torchaudio)
COPY demucs/requirements.txt /tmp/demucs_requirements.txt
RUN pip install --no-cache-dir -r /tmp/demucs_requirements.txt

# Install Onsets-and-Frames dependencies
COPY onsets-and-frames/requirements.txt /tmp/onsets_requirements.txt
RUN pip install --no-cache-dir -r /tmp/onsets_requirements.txt

# Copy repo and install Demucs in editable mode
COPY demucs /workspace/demucs
RUN pip install --no-cache-dir -e /workspace/demucs

WORKDIR /workspace
