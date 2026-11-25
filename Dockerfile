FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m appuser && mkdir -p /state && chown appuser:appuser /state
USER appuser
EXPOSE 8080
CMD ["python", "secure_web.py", "--host", "0.0.0.0", "--database", "/state/unimonitor.db", "--registry", "/state/model_runs"]
