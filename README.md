# Local Agent

Local Agent is a small, locally hosted ReAct-style agent that runs against Ollama and gives the model access to tools for web search, file handling, system inspection, Docker queries, and a local knowledge base. It is designed to run fully on a local machine without depending on a hosted API.

This project is currently configured around a Qwen model and works best with that family of models.

## Features

- ReAct-style reasoning loop with tool use
- Web search via DuckDuckGo
- Wikipedia and IMDb lookups
- Weather lookups via wttr.in
- Workspace file read/write tools
- Host system and process inspection
- Docker container inspection
- Sandboxed command execution in a temporary container
- Persistent chat history and a local Chroma vector database

## Requirements

- Python 3.10+
- Docker and Docker Compose (recommended for the full setup)
- Ollama running locally or via Docker Compose
- Sufficient RAM for your chosen model; 4 GB minimum is usually workable, with more recommended

## Python dependencies

The Python packages are listed in requirements.txt:

- ollama
- ddgs
- wikipedia
- requests
- prompt_toolkit
- docker
- beautifulsoup4
- lxml
- psutil
- pymysql
- chromadb

Install them with:

```bash
python3 -m pip install -r requirements.txt
```

## Quick start with Docker Compose

1. Clone the repository and change into it:

```bash
git clone <repo-url>
cd local-agent
```

2. Start the services:

```bash
docker compose up -d --build
```

3. Attach to the agent container:

```bash
docker attach terminal_agent
```

4. Enter a prompt such as:

```text
What is the weather in London?
Read the workspace files and summarize them.
Search the web for recent updates about local AI tools.
```

The compose file starts:

- an Ollama service on port 11434
- the agent container with access to the workspace and memory folders

## Running locally without Docker

If you prefer to run the agent directly on the host:

1. Make sure Ollama is running and reachable.
2. Set the Ollama host if needed:

```bash
export OLLAMA_HOST=http://localhost:11434
```

3. Start the agent:

```bash
python3 agent.py
```

## Configuration

### Ollama

The agent uses the Ollama host from the environment variable OLLAMA_HOST. In Docker Compose this is already set to the service name:

```yaml
environment:
  - OLLAMA_HOST=http://ollama:11434
```

If you run the agent outside Docker, set it to your local Ollama endpoint instead.

### Model selection

The model name is defined in agent.py:

```python
MODEL_NAME = "qwen2.5-coder:3b"
```

Change it if you want to use a different local model available in your Ollama installation. The project is currently tuned around a Qwen model and is expected to perform best with that family.

### Paths and storage

The agent uses these defaults:

- MEMORY_DIR: /app/memory inside the container, or the environment override you provide
- WORKSPACE_DIR: /app/workspace inside the container

The local folders are mounted in docker-compose.yml:

```yaml
volumes:
  - ./memory:/app/memory
  - ./workspace:/app/workspace
```

### Optional database settings

The agent includes a read-only MariaDB helper. If you want to use it, configure these environment variables before running:

```bash
export DB_HOST=...
export DB_USER=...
export DB_PASS=...
export DB_NAME=...
```

## Project layout

```text
local-agent/
├── agent.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── memory/
│   ├── chat_history.json
│   └── cmd_history.txt
├── workspace/
├── ollama_data/
└── README.md
```

## Built-in commands

Inside the interactive prompt, you can use:

- /think to toggle verbose reasoning output
- /wipe to clear chat memory
- /wipe_kb to reset the knowledge base
- /copy to copy the last agent response to the clipboard
- exit or quit to end the session

## Troubleshooting

If the agent does not start:

```bash
docker compose ps
docker compose logs agent
docker compose logs ollama
```

If the model is not present, pull it from Ollama manually:

```bash
docker compose exec ollama ollama pull qwen2.5-coder:3b
```

If you are running outside Docker, check that Ollama is available at the address set in OLLAMA_HOST.

## License

This project is provided as-is for local experimentation and development.
