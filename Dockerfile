# NoteMind container image: runs the Streamlit app on port 7860.
# Works on Hugging Face Docker Spaces and any other Docker host.
#
#   docker build -t notemind .
#   docker run -p 7860:7860 -e OPENROUTER_API_KEY=... -e NOTEMIND_PROVIDER=openrouter notemind
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run containers as user ID 1000; create that user and
# install everything into its home so no step needs root at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app/src
WORKDIR $HOME/app

# 1) CPU-only PyTorch first. Installed from PyPI on Linux, torch pulls ~3 GB of
#    CUDA libraries; the CPU wheel is a fraction of that and builds much faster.
RUN pip install --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2) The rest of the dependencies. This layer is cached until requirements.txt
#    changes, so code-only changes rebuild in seconds.
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# 3) Bake the embedding model into the image so the app never downloads it at
#    startup (faster cold starts, works even if the Hub is slow).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

# 4) Application code (see .dockerignore for what is left out).
COPY --chown=user . .

# Streamlit settings for running behind a proxy/iframe such as Hugging Face:
# XSRF protection relies on a cookie that browsers block in third-party
# iframes, which breaks file uploads there. Remove this line on other hosts.
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
