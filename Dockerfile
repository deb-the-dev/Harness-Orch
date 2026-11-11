# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Open Policy Agent (OPA) binary for local sidecar or out-of-process hook evaluation
ARG OPA_VERSION=1.2.0
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL -o /usr/local/bin/opa "https://openpolicyagent.org/downloads/v${OPA_VERSION}/opa_linux_${ARCH}_static" && \
    chmod +x /usr/local/bin/opa

COPY pyproject.toml README.md ./
COPY src ./src
COPY policies ./policies
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e ".[test]"

EXPOSE 8080

CMD ["uvicorn", "harness.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
