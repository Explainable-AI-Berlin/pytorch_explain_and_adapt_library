# PEAL web demo image for the NVIDIA DGX Spark (GB10, aarch64).
#
#   docker build -f deploy/spark/Dockerfile -t peal-web .
#
# Base: NVIDIA's aarch64 PyTorch container (CUDA 13 / Blackwell support; the
# PyPI CUDA wheels for linux/aarch64 also work but the NGC image is the tested
# combination on the Spark). Everything the RAE generator needs on top is
# installed by tools/install_rae.py, which copies the repository's RAEv2 fork
# (packages/peal-xai-rae, CC BY-NC 4.0: you accept its licence by building this
# image) to $PEAL_RAEV2_DIR.
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PEAL_RUNS=/data/peal_runs PEAL_DATA=/data/datasets \
    PEAL_WEB_JOBS=/data/web_jobs PEAL_RAEV2_DIR=/opt/RAEv2 \
    HF_HOME=/data/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/peal
COPY . .
# The package with its extras, not requirements.txt: that file pins the
# research environment (e.g. numba 0.57, which has no Python 3.12 build) and
# does not install on this base image. torch/torchvision stay the base image's
# builds (the constraint file); OpenAI CLIP and the RAE extras come on top.
RUN pip list --format=freeze | grep -iE '^(torch|torchvision)==' > /tmp/keep-torch.txt \
    && pip install --no-cache-dir --ignore-requires-python -c /tmp/keep-torch.txt -e ".[web,onnx,clip]" \
       "clip @ git+https://github.com/openai/CLIP.git" lightning lpips lmdb \
    && python tools/install_rae.py --dir /opt/RAEv2 --yes

EXPOSE 8080
CMD ["uvicorn", "peal.web.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
