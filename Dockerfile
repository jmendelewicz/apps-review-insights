
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

# Run from /app so that "from src.API.container import ..." resolves correctly
CMD ["uvicorn", "src.API.review_api:app", "--host", "0.0.0.0", "--port", "7860"]
