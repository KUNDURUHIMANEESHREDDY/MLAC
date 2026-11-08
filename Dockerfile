FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN groupadd --system --gid 10001 mlac \
    && useradd --system --uid 10001 --gid mlac --home-dir /nonexistent mlac
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY schemas ./schemas
COPY scripts ./scripts
RUN python -m pip install --upgrade pip \
    && pip install '.[api,production]'

RUN mkdir -p /var/lib/mlac && chown 10001:10001 /var/lib/mlac
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "mlac.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=172.30.0.2"]
