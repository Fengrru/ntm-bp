# Dockerfile for NTM-BP submission package
# Build: docker build -t ntm-bp:v1.0-submission .
# Run:   docker run --rm ntm-bp:v1.0-submission

FROM python:3.11-slim

# Install system deps (for any native wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and install in editable mode
COPY ntm-bp/ ./ntm-bp/
RUN pip install --no-cache-dir -e ./ntm-bp

# Copy all data files (generated datasets + frozen cache)
COPY ntm-bp/data/ ./data/

# Default command: run the main three-way comparison (Figure 1)
# Override with other scripts as needed:
#   docker run --rm ntm-bp:v1.0-submission python scripts/experiment_keyword_baseline.py
#   docker run --rm -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY ntm-bp:v1.0-submission python scripts/experiment_sealed_test.py
ENTRYPOINT ["python", "scripts/run_main_eval.py"]