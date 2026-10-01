FROM python:3.12-slim

LABEL org.opencontainers.image.title="Cross Section Studio" \
      org.opencontainers.image.authors="Andrew Liu, Ecoventure" \
      org.opencontainers.image.vendor="Ecoventure" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary" \
      org.opencontainers.image.description="Created by Andrew Liu, Ecoventure, 2026. Copyright (c) 2026 Andrew Liu, Ecoventure. All rights reserved."

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgeos-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY VERSION ./
COPY gwm_reference ./gwm_reference
COPY advantage_p2_reference ./advantage_p2_reference
COPY data ./data
COPY docs ./docs
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
