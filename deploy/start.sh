#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${PORT:-10000}"
export ATLAS_ASSET_BIND_HOST="${ATLAS_ASSET_BIND_HOST:-127.0.0.1}"
export ATLAS_ASSET_PORT="${ATLAS_ASSET_PORT:-8765}"
export ATLAS_URL_PREFIX="${ATLAS_URL_PREFIX:-/atlas}"

envsubst '${PORT}' < /app/deploy/nginx.conf.template > /etc/nginx/conf.d/default.conf
nginx -t

streamlit run app.py \
    --server.address=127.0.0.1 \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

cleanup() {
    kill "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
    wait "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$STREAMLIT_PID" "$NGINX_PID"
