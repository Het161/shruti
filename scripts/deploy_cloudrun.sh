#!/usr/bin/env bash
# Deploy SHRUTI to Google Cloud Run.
#
# Cloud Run replaced the original Hugging Face Docker Space target, which now returns 402:
# HF requires a PRO subscription for Docker Spaces on cpu-basic. Cloud Run gives real vCPUs, a US
# region, and runs this repo's Dockerfile unmodified — at this traffic volume it stays inside the
# always-free tier.
#
# Prerequisites (one-time, and they need YOUR Google account — I cannot do these for you):
#
#   gcloud auth login
#   gcloud config set project <PROJECT_ID>
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
#
# Then:
#   ./scripts/deploy_cloudrun.sh
#
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-shruti}"
ARTIFACT_REPO="${SHRUTI_ARTIFACT_REPO:-Het0812/shruti-artifacts}"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "ERROR: no GCP project set. Run: gcloud config set project <PROJECT_ID>" >&2
  exit 1
fi

# Keys are read from .env and pushed as Cloud Run env vars. They are never baked into the image —
# an image layer is readable by anyone who can pull it.
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

echo "project ${PROJECT} · region ${REGION} · service ${SERVICE}"
echo "artifacts ${ARTIFACT_REPO}"

# --- memory sizing -----------------------------------------------------------------------
# 2Gi is deliberate, and measured rather than guessed:
#   embedding matrix (310,582 x 256 float32, RAM-resident, not mmapped)   318 MB
#   potion static model                                                  ~512 MB
#   HNSW graph (RAM-resident)                                            ~330 MB
#   passage text (Arrow, contiguous UTF-8)                               ~200 MB
#   BM25 index                                                           ~200 MB
#   Python + numpy + framework                                           ~300 MB
#                                                                        --------
#                                                                        ~1.9 GB
# The indexes are RAM-resident on purpose: memory-mapping them produced a 1,728ms P100 from
# first-touch page faults. See docs/BUILD_LOG.md. Do not "optimise" this back down.
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"

gcloud run deploy "${SERVICE}" \
  --source . \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --timeout 300 \
  --concurrency 8 \
  --max-instances 2 \
  --min-instances 0 \
  --port 8080 \
  --set-env-vars "SHRUTI_ARTIFACT_DIR=/app/artifacts,SHRUTI_EMBED_LANE=fast,SHRUTI_SEARCH_MODE=hnsw,SHRUTI_WARMUP_QUERIES=20,SARVAM_API_KEY=${SARVAM_API_KEY:-},CEREBRAS_API_KEY=${CEREBRAS_API_KEY:-},GROQ_API_KEY=${GROQ_API_KEY:-}"

URL="$(gcloud run services describe "${SERVICE}" --project "${PROJECT}" --region "${REGION}" --format='value(status.url)')"

echo
echo "deployed: ${URL}"
echo
echo "next:"
echo "  1. set the SHRUTI_URL repo variable so CI can find it:"
echo "       gh variable set SHRUTI_URL --body '${URL}'"
echo "  2. benchmark it from a US runner (not your laptop — that measures the Pacific):"
echo "       gh workflow run benchmark"
echo
echo "checking health…"
curl -sS --max-time 180 "${URL}/api/health" | head -c 600
echo
