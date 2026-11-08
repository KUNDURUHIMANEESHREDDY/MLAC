FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/mlac-worker/lib

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.5.1 'cryptography>=42' 'transformers==4.48.3' 'safetensors==0.5.3'

RUN mkdir -p /opt/mlac-worker/lib/mlac
COPY src/mlac/__init__.py /opt/mlac-worker/lib/mlac/__init__.py
COPY src/mlac/ir.py /opt/mlac-worker/lib/mlac/ir.py
COPY src/mlac/registry.py /opt/mlac-worker/lib/mlac/registry.py
COPY src/mlac/transformer.py /opt/mlac-worker/lib/mlac/transformer.py
COPY src/mlac/attestation.py /opt/mlac-worker/lib/mlac/attestation.py
COPY src/mlac/lm_evaluation.py /opt/mlac-worker/lib/mlac/lm_evaluation.py
COPY worker/evaluate /opt/mlac-worker/evaluate
COPY worker/evaluate_lm /opt/mlac-worker/evaluate-lm

RUN chmod 0555 /opt/mlac-worker/evaluate /opt/mlac-worker/evaluate-lm
USER 65532:65532
