FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Set Hugging Face cache environment variables (optional, for consistency)
# In Docker, the default cache location (/root/.cache/huggingface) works fine
# but you can customize it if needed
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Expose Jupyter port
EXPOSE 8888

# Entrypoint
# Option 1: Sans token (simple pour dev local) - Décommentez pour utiliser
# CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]

# Option 2: Token fixe (recommandé pour dev local) - Actuellement utilisé
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token='illuin2024'"]
