# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

WORKDIR /workspace

COPY stage3/search/python_service /workspace

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8091

CMD ["python", "app.py"]
