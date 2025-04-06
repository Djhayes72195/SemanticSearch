FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ENV PYTHONPATH=/app


RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY prod-requirements.txt .
RUN pip install --no-cache-dir -r prod-requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000
CMD ["uvicorn", "SearchApp.api:app", "--host", "0.0.0.0", "--port", "8080"]
