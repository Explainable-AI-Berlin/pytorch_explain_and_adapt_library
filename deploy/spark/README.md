# PEAL web demo on a DGX Spark

The demo is `peal/web`: a FastAPI app with one worker thread that runs
`run_didae.py` for one uploaded job at a time. It needs the published RAE
weights and the MSAE dictionary (downloaded from Hugging Face on first use)
and nothing else from the research runs.

## 1. Publish the generator weights (once, from the training host)

    python tools/export_rae_weights.py \
        --base_path $PEAL_RUNS/imagenet/rae_clip \
        --stage2_checkpoint $PEAL_RUNS/imagenet/rae_clip/stage2_slim/ep-0000043_ema.pt \
        --out /tmp/peal-rae-clip-imagenet \
        --upload <org>/peal-rae-clip-imagenet

The folder holds only `decoder.pt`, `stats.pt` and `stage2_ema.pt` (about
6.5 GB in fp32; `--dtype bf16` halves it). A generator config then uses
`weights: hf://<org>/peal-rae-clip-imagenet`; the demo reads it from
`$PEAL_RAE_WEIGHTS`.

## 2. Build and run on the Spark

    git clone https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library.git && cd pytorch_explain_and_adapt_library
    sed -i 's#hf://CHANGE_ME/peal-rae-clip-imagenet#hf://<org>/peal-rae-clip-imagenet#' deploy/spark/docker-compose.yaml
    mkdir -p /data/peal
    docker compose -f deploy/spark/docker-compose.yaml up -d --build
    docker compose -f deploy/spark/docker-compose.yaml logs -f

Open http://<spark>:8080. The first job downloads the weights (RAE ~6.5 GB,
MSAE ~50 MB, CLIP ViT-L/14 ~900 MB) into `/data`.

The Spark is aarch64: the base image is NVIDIA's aarch64 PyTorch container,
`onnxruntime` is the CPU wheel (the uploaded classifier is converted to torch
with onnx2torch anyway, so it runs on the GPU), and the RAEv2 fork
(`packages/peal-xai-rae`) is installed by `tools/install_rae.py` inside the image (CC BY-NC 4.0: the demo is
non-commercial and says so on the page).

## 3. Expose it

The app has no authentication or rate limiting beyond the upload caps. Put it
behind a TLS reverse proxy (Caddy, nginx) or a Cloudflare Tunnel / Tailscale
Funnel, and keep `PEAL_WEB_MAX_ZIP_MB` / `PEAL_WEB_MAX_ONNX_MB` at what the
disk can take. Job directories under `/data/web_jobs` are never deleted
automatically; a cron that removes finished jobs older than N days is the
retention policy.

## Without docker

    pip install -r requirements-web.txt
    python tools/install_rae.py
    export PEAL_RAE_WEIGHTS=hf://<org>/peal-rae-clip-imagenet PEAL_WEB_JOBS=/data/web_jobs PEAL_RUNS=/data/peal_runs
    uvicorn peal.web.app:app --host 0.0.0.0 --port 8080

## Smoke test without a GPU or weights

    PEAL_WEB_NO_WORKER=1 uvicorn peal.web.app:app --port 8080

accepts uploads and creates jobs (they stay queued); the tests in
`tests/web` cover the API, the ingest, the ONNX conversion and the feedback
round trip.
