#!/bin/sh
set -eu

: "${PORT:=10000}"
export PORT

envsubst '$PORT' < /app/deploy/nginx.conf.template > /etc/nginx/nginx.conf

streamlit run app.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &
STREAMLIT_PID=$!

cleanup() {
  kill "$STREAMLIT_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

nginx -g 'daemon off;'
