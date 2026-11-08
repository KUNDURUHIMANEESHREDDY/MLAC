FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/mlac-worker/lib

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

RUN mkdir -p /opt/mlac-worker/lib/mlac
COPY src/mlac/__init__.py /opt/mlac-worker/lib/mlac/__init__.py
COPY src/mlac/ir.py /opt/mlac-worker/lib/mlac/ir.py
COPY src/mlac/registry.py /opt/mlac-worker/lib/mlac/registry.py
COPY src/mlac/transformer.py /opt/mlac-worker/lib/mlac/transformer.py
COPY worker/evaluate /opt/mlac-worker/evaluate

RUN chmod 0555 /opt/mlac-worker/evaluate
USER 65532:65532
