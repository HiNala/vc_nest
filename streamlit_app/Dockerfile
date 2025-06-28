FROM python:3.12-slim

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir \
    streamlit==1.35.0 \
    requests==2.31.0 \
    sseclient-py==1.8.0

# Copy Streamlit app
COPY MovieDebate.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "MovieDebate.py", "--server.address", "0.0.0.0", "--server.port", "8501"] 