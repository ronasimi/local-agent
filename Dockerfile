FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

CMD ["python3", "agent.py"]
