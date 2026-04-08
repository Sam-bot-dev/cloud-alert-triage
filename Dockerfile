FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements-pinned.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Python-based healthcheck — curl is not available in slim images
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

USER 1000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
