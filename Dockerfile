FROM python:3.12-slim

WORKDIR /srv

# vLLM needs a CUDA-capable base for GPU inference; for a GPU image,
# start from nvidia/cuda:12.4.1-runtime-ubuntu22.04 instead and install
# Python there. This slim image is fine for building / CPU smoke tests.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
    LOAD_MODEL_ON_STARTUP=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
