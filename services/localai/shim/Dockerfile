FROM python:3.10-slim
WORKDIR /app
COPY services/localai/shim/app.py /app/app.py
RUN pip install fastapi uvicorn httpx redis
ENV LOCALAI_CORE_URL=http://localai:8080
ENV REDIS_URL=redis://redis:6379/0
ENV REDIS_DOMAIN_CONFIG_KEY=localai:domains:config
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
