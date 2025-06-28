FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY cli.py .
COPY scripts/ ./scripts/
COPY data/movies.csv .
COPY demo_profiles.json .

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Default command runs the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 