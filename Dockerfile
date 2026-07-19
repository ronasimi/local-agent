FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Start the Streamlit web server
CMD ["python3", "-m", "streamlit", "run", "web_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
