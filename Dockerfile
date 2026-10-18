# Dockerfile for the NTM-BP reproducibility package.
# Build from the repository root:
#   docker build -t ntm-bp:v1.0-submission .
# Default (zero API, offline): verify the paper's LLM-derived tables.
#   docker run --rm ntm-bp:v1.0-submission
# Other examples:
#   docker run --rm ntm-bp:v1.0-submission python scripts/run_main_eval.py
#   docker run --rm -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY ntm-bp:v1.0-submission \
#       python scripts/experiment_sealed_test.py

FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r requirements-repro.txt

# Offline reproduction from the frozen prompt-hash cache; no API key needed.
ENTRYPOINT ["python", "scripts/reproduce_paper_tables.py"]
