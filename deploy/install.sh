#!/usr/bin/env bash
# Start Cue v1 on a fresh NVIDIA GPU machine (Ubuntu/Debian: EC2 g4dn, GCP T4, any provider).
# Run it as root, or paste it into "user data" (EC2) / "startup script" (GCP) when launching the VM.
#   CUE_API_KEY=your-secret ./install.sh      # the key is generated and printed if you don't set one
set -euo pipefail

DIR=/opt/cue
KEY="${CUE_API_KEY:-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')}"

command -v nvidia-smi >/dev/null || { echo "No NVIDIA driver found. Use a GPU image (e.g. AWS Deep Learning AMI, GCP Deep Learning VM) or install the driver first." >&2; exit 1; }

if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi

mkdir -p "$DIR/models" && cd "$DIR"
cat > docker-compose.yml <<YML
services:
  cue:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    restart: unless-stopped
    ports: ["8090:8090"]
    volumes: ["$DIR/models:/models"]
    environment:
      LLAMA_CACHE: /models
      HF_HUB_CACHE: /models
      LLAMA_API_KEY: "$KEY"
    command: >
      --hf-repo Pype/cue-v1 --hf-file cue-v1-q8_0.gguf -c 8192 -np 4 -ngl 999 --jinja --host 0.0.0.0 --port 8090
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
YML
chmod 600 docker-compose.yml
docker compose up -d

IP=$(curl -fsS --max-time 3 https://checkip.amazonaws.com 2>/dev/null || hostname -I | cut -d' ' -f1)
echo
echo "Cue is starting (the first start downloads the 8 GB model: docker compose -f $DIR/docker-compose.yml logs -f)."
echo "  CUE_URL=http://${IP}:8090/v1"
echo "  CUE_API_KEY=$KEY"
echo "Open port 8090 only to your agent's IP / security group."
