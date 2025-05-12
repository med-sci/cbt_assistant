FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# add the cbt_assistant package installation

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]