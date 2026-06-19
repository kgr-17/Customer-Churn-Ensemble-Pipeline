FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KAGGLE_CONFIG_DIR=/kaggle \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

EXPOSE 8888

CMD ["sh", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=${JUPYTER_TOKEN} --ServerApp.allow_origin='*' --ServerApp.allow_remote_access=True --allow-root"]

