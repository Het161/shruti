#!/usr/bin/env bash
# SHRUTI — one-shot provisioning for an Oracle Cloud Always Free ARM instance.
#
# Run this ON THE VM, once, as the default `ubuntu` user:
#
#     curl -fsSL https://raw.githubusercontent.com/Het161/shruti/main/deploy/oracle/setup.sh | bash -s -- \
#         --domain shruti-het.duckdns.org --email you@example.com
#
# What it does, in order:
#   1. opens the firewall (Oracle images ship with iptables closed AND a cloud-level security list)
#   2. installs Docker
#   3. clones the repo and builds the image for arm64
#   4. fetches corpus artifacts and BUILDS the derived indexes locally
#   5. installs nginx with WebSocket proxying and a Let's Encrypt certificate
#   6. registers a systemd unit so the container survives reboots
#
# Why the indexes are built here rather than downloaded: they are ~450 MB of derived data that a
# 4-core ARM box regenerates in about a minute, and the upload from a laptop on a ~3 MB/s Indian
# link failed repeatedly. Compute in the datacentre beats bytes over a bad link — the same reason
# the Modal deployment does it this way.
#
# ARM notes, since this is aarch64 and not x86:
#   - python:3.13-slim, onnxruntime, usearch and pyarrow all publish aarch64 wheels.
#   - The AVX512-VNNI int8 ONNX exports do NOT run here. app/stages/embed.py and rerank.py already
#     fall back to fp32, so this degrades in speed rather than breaking. Expect the reranker to be
#     slower than the numbers measured on x86.

set -euo pipefail

DOMAIN=""
EMAIL=""
REPO="https://github.com/Het161/shruti.git"
ARTIFACT_REPO="${SHRUTI_ARTIFACT_REPO:-Het0812/shruti-artifacts}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --email)  EMAIL="$2";  shift 2 ;;
    --repo)   REPO="$2";   shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$DOMAIN" ]] && { echo "ERROR: --domain required (e.g. shruti-het.duckdns.org)" >&2; exit 1; }
[[ -z "$EMAIL"  ]] && { echo "ERROR: --email required (Let's Encrypt expiry notices)" >&2; exit 1; }

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }

# --- 1. firewall ----------------------------------------------------------------------
# Oracle's Ubuntu image ships with a REJECT rule in iptables that silently blackholes 80/443.
# This is the single most common reason a correctly-deployed Oracle VM appears dead.
say "Opening ports 80/443 in the instance firewall"
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || {
  sudo apt-get update -qq && sudo apt-get install -y iptables-persistent
  sudo netfilter-persistent save
}
echo "NOTE: you must ALSO open 80/443 in the VCN Security List in the Oracle console."

# --- 2. docker ------------------------------------------------------------------------
say "Installing Docker"
sudo apt-get update -qq
sudo apt-get install -y ca-certificates curl gnupg git nginx
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

# --- 3. source ------------------------------------------------------------------------
say "Cloning SHRUTI"
sudo mkdir -p /opt/shruti && sudo chown "$USER:$USER" /opt/shruti
if [[ -d /opt/shruti/.git ]]; then git -C /opt/shruti pull --ff-only; else git clone "$REPO" /opt/shruti; fi
cd /opt/shruti

# --- 4. artifacts ---------------------------------------------------------------------
say "Fetching corpus and building indexes (this takes ~3 min on 4 ARM cores)"
mkdir -p /opt/shruti/artifacts
sudo docker run --rm -v /opt/shruti/artifacts:/artifacts python:3.13-slim bash -c "
  pip install -q huggingface-hub pyarrow numpy usearch bm25s PyStemmer &&
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('$ARTIFACT_REPO', repo_type='dataset', local_dir='/artifacts',
                  allow_patterns=['*.parquet','*.npy','*.json'])
print('source artifacts downloaded')
PY
"
# The derived indexes are built by the app image itself so they are produced by exactly the code
# that will serve them — a mismatch between index format and reader is a silent, confusing failure.
say "Building the application image for arm64"
sudo docker build -t shruti:latest /opt/shruti

# --user root because the image runs as uid 1000 but the host-mounted artifacts directory is owned
# by the VM's default user; without this the index write fails with a permission error that reads
# like a code bug.
sudo docker run --rm --user root -v /opt/shruti/artifacts:/artifacts -w /app shruti:latest python - <<'PY'
import time, numpy as np, pyarrow.parquet as pq
from pathlib import Path
from app.stages.dense import DenseIndex
from app.stages.lexical import LexicalIndex
d = Path("/artifacts")
if not (d / "hnsw.usearch").exists():
    t = time.perf_counter(); DenseIndex.build(np.load(d / "embeddings.npy")).save(d / "hnsw.usearch")
    print(f"HNSW built in {time.perf_counter()-t:.1f}s")
if not (d / "bm25").exists():
    texts = pq.read_table(d / "passages.parquet", columns=["text"]).column("text").to_pylist()
    t = time.perf_counter(); LexicalIndex.build(texts).save(str(d / "bm25"))
    print(f"BM25 built in {time.perf_counter()-t:.1f}s")
PY

# --- 5. run ---------------------------------------------------------------------------
say "Starting SHRUTI"
sudo docker rm -f shruti 2>/dev/null || true
sudo docker run -d --name shruti --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -v /opt/shruti/artifacts:/app/artifacts:ro \
  --env-file /opt/shruti/.env \
  -e SHRUTI_ARTIFACT_DIR=/app/artifacts \
  shruti:latest

# --- 6. nginx + TLS -------------------------------------------------------------------
say "Configuring nginx (with WebSocket upgrade for the voice path)"
sudo tee /etc/nginx/sites-available/shruti >/dev/null <<NGINX
server {
    listen 80;
    server_name $DOMAIN;

    # The voice path is a WebSocket. Without the Upgrade/Connection headers nginx silently
    # downgrades it to a normal request and the microphone never connects — which looks like a
    # broken app rather than a broken proxy.
    location /ws/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/shruti /etc/nginx/sites-enabled/shruti
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

say "Requesting a Let's Encrypt certificate"
# getUserMedia refuses to run on http:// or on a self-signed cert, so a real certificate is not
# optional here — it is the difference between a working microphone and a dead button.
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect

say "Done"
echo "  https://$DOMAIN"
echo
echo "  health : curl https://$DOMAIN/api/health"
echo "  logs   : sudo docker logs -f shruti"
echo "  update : cd /opt/shruti && git pull && sudo docker build -t shruti:latest . && sudo docker restart shruti"
