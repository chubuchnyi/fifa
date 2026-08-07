# pitch3d — reconstruction container for an Ada (sm_89) GPU such as the RTX 4080.
#
# Mirrors scripts/cloud_setup.sh's REUSE_SYSTEM_TORCH path rather than reinventing it:
# this base image already ships the exact pinned pair (torch 2.6.0+cu124 /
# torchvision 0.21.0), so we hold it with a constraints file and let the extras
# install around it. Installing torch again would only risk pip resolving a
# CPU/mismatched wheel over it — the lesson recorded in cloud_setup.sh step 4.
#
# Weights are NOT baked in (~49 GB, several license-gated). Mount them:
#   -v /path/to/models:/models -e PITCH3D_MODELS=/models
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/hf

# ffmpeg/ffprobe: required by the real FFmpegIngestor for --clip and every mp4 stitch.
# The libx*/libgl set is what headless Blender needs (scripts/pod_ensure_blender.sh:28-38);
# opencv needs libgl1 + libglib2.0-0 too.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates \
        libgl1 libglib2.0-0 \
        libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Hold the image's CUDA torch pair so the torch==2.6.0-pinned extras cannot pull
# a wheel over it (rfdetr drags in an unpinned torchvision — cloud_setup.sh:86-91).
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /tmp/constraints.txt

# Dependency layer first: copy only what pip reads, so edits to src/ don't
# invalidate the (slow) install layer.
COPY pyproject.toml README.md ./
# chumpy==0.70 (pulled by the `hmr` extra) is a 2019 package whose setup.py imports
# `pip` itself. Modern pip builds in an isolated env that has no pip, so it dies with
# "ModuleNotFoundError: No module named 'pip'". Install it first with isolation off —
# then the extras' chumpy requirement is already satisfied and pip won't retry it.
RUN mkdir -p src/pitch3d && touch src/pitch3d/__init__.py \
    && python -m pip install -U pip wheel setuptools \
    && python -m pip install --no-build-isolation chumpy==0.70 \
    && python -m pip install -e ".[cv,hmr,ball,export,mcp,dev]" -c /tmp/constraints.txt

# Runtime deps the project uses but does not declare in pyproject.toml:
#   fastapi/uvicorn/pydantic — poseannot Studio (poseannot/app.py)
#   shapely                  — PnLCalib      (docs/runpod-agent-setup.md:159)
#   hydra-core               — WASB          (docs/runpod-agent-setup.md:170)
#   gdown, huggingface_hub   — weight staging
RUN python -m pip install -c /tmp/constraints.txt \
        fastapi uvicorn pydantic shapely hydra-core gdown "huggingface_hub[cli]"

COPY . .

# `pip install -e` recorded the package before src/ existed; re-run so the
# editable install points at the real tree.
RUN python -m pip install -e . --no-deps -c /tmp/constraints.txt

ENV PYTHONPATH=/app/src
CMD ["bash"]
