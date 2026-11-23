FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx gettext-base ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-stage*.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN chmod +x /app/deploy/start.sh \
    && rm -f /etc/nginx/sites-enabled/default

EXPOSE 10000

CMD ["/app/deploy/start.sh"]
