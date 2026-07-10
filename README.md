# 🤖 Local AI Agent

A self-contained, locally-running autonomous AI agent powered by [Ollama](https://ollama.ai) and built on the **ReAct (Reason + Act)** framework. This agent can search the web, access system information, manage files, and autonomously solve problems—all running entirely on your machine with no external API dependencies.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **🧠 Autonomous Reasoning**: Uses the ReAct protocol to think through problems step-by-step
- **🌐 Web Search**: Powered by DuckDuckGo for real-time information retrieval
- **📚 Knowledge Retrieval**: Wikipedia integration for comprehensive background information
- **🎬 Entertainment Search**: IMDb lookups for movies, shows, and actors
- **🌤️ Weather Data**: Real-time weather information for any location
- **📁 File Operations**: Read and write files in the workspace
- **💻 System Access**: Query host system information and performance metrics
- **💾 Memory Management**: Persistent chat history and command logging
- **🐳 Containerized**: Complete Docker setup with Ollama integration
- **⚡ Local & Private**: Everything runs locally—no data leaves your machine

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM (recommended 8GB+)
- GPU support optional (iGPU enabled in compose file)

### Installation

1. **Clone or download the repository**
   ```bash
   cd local-agent
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

   This will:
   - Start the Ollama service
   - Build and run the agent container
   - Set up persistent volumes for memory and workspace

3. **Interact with the agent**
   ```bash
   docker attach terminal_agent
   ```

4. **Type your queries** and press Enter!
   ```
   What's the weather in London?
   Search for information about the 2024 Olympics
   Tell me about the latest AI developments
   ```

---

## 🛠️ Available Tools

The agent has access to the following tools:

| Tool | Purpose | Example |
|------|---------|---------|
| `search_ddg` | Search the live internet | "Find the latest news about AI" |
| `search_wikipedia` | Access Wikipedia knowledge | "Tell me about quantum computing" |
| `search_imdb` | Look up movies, shows, actors | "Who starred in Inception?" |
| `get_weather` | Get current weather conditions | "What's the weather in Tokyo?" |
| `read_file` | Read workspace files | "Read my notes.txt" |
| `write_file` | Save content to files | "Save the results to output.txt" |
| `read_system_proc` | Access system information | "How much RAM am I using?" |
| `get_system_time` | Check current date/time | "What time is it?" |

---

## 📋 Configuration

### Environment Variables
Edit `docker-compose.yml` to customize:

```yaml
environment:
  - OLLAMA_HOST=http://ollama:11434      # Ollama service address
  - TZ=America/Toronto                    # Timezone
  - OLLAMA_CONTEXT_LENGTH=8192            # LLM context window
  - OLLAMA_VULKAN=1                       # GPU acceleration (AMD)
  - OLLAMA_IGPU_ENABLE=1                  # iGPU support
```

### Model Selection
Change the model in `agent.py`:
```python
MODEL_NAME = "qwen2.5-coder:7b"  # Replace with your preferred model
```

**Available models:**
- `qwen2.5-coder:7b` (Current - good for coding tasks)
- `llama2:7b`, `llama2:13b` (General purpose)
- `mistral:7b` (Fast & efficient)
- `neural-chat:7b` (Chat optimized)

---

## 📁 Project Structure

```
local-agent/
├── agent.py                 # Main agent script with ReAct loop
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-container orchestration
├── memory/
│   ├── chat_history.json   # Persistent conversation history
│   └── cmd_history.txt     # Command history log
├── workspace/              # Shared workspace for file operations
├── ollama_data/            # Ollama model cache & keys
└── README.md              # This file
```

---

## 🧠 How It Works

The agent operates on the **ReAct (Reason + Act)** protocol:

```
1. THOUGHT → Agent reasons about the problem
2. ACTION  → Selects an appropriate tool
3. OBSERVATION → Receives tool results
4. REPEAT → Cycles until reaching a conclusion
5. FINAL ANSWER → Delivers the response to the user
```

**Example interaction:**
```
User: What's the weather in Paris?

Agent Thought: The user wants weather information for Paris. 
I should use the get_weather tool.

Action: get_weather
Action Input: Paris

Observation: Paris: +18°C, Partly cloudy, 65% humidity

Final Answer: The weather in Paris is 18°C with partly cloudy 
conditions and 65% humidity.
```

---

## 📦 Installation & Dependencies

### Python Dependencies
- `ollama` - Ollama Python client
- `ddgs` - DuckDuckGo Search
- `wikipedia` - Wikipedia API
- `requests` - HTTP library
- `prompt_toolkit` - Interactive CLI with history

### Docker Images
- `python:3.10-slim` - Lightweight Python runtime
- `ollama/ollama:latest` - Ollama service

---

## 🎯 Usage Examples

### Example 1: Web Research
```
> Research the latest developments in quantum computing
  and save a summary to quantum_summary.txt
```

### Example 2: System Diagnostics
```
> How much memory is the system currently using?
  Check CPU info and tell me about the processor.
```

### Example 3: Entertainment
```
> Search IMDb for "The Matrix" and tell me about the cast.
```

### Example 4: Workspace Management
```
> Read the file notes.txt from my workspace and summarize it.
> Save the results to output.txt
```

---

## 🐛 Troubleshooting

### Agent not responding
```bash
# Check container status
docker ps

# View agent logs
docker logs terminal_agent

# View Ollama logs
docker logs ollama_service
```

### Model download stuck
```bash
# Check Ollama service
docker logs ollama_service

# Manual model pull
docker exec ollama_service ollama pull qwen2.5-coder:7b
```

### Memory issues
- Increase Docker memory: Edit `docker-compose.yml` or Docker Desktop settings
- Reduce context length: Lower `OLLAMA_CONTEXT_LENGTH`
- Use a smaller model: Switch to `mistral:7b` or `neural-chat:7b`

### GPU not being used
- Check device support: Verify `/dev/kfd` and `/dev/dri` availability
- Update Ollama: `docker pull ollama/ollama:latest`
- Check Docker GPU settings in `docker-compose.yml`

---

## 🚀 Advanced Configuration

### Adding New Tools
Edit `agent.py` to add custom tools:

```python
def custom_tool(query: str) -> str:
    """Your custom tool description."""
    # Implementation
    return result

tools_map['custom_tool'] = custom_tool
```

### Persistent Memory
Chat history is automatically saved to `memory/chat_history.json`. To clear history:
```bash
docker exec terminal_agent rm /app/memory/chat_history.json
```

### Verbose Mode
The agent includes a verbose mode for debugging. Modify `main()` to enable:
```python
verbose_mode = True
```

---

## 📊 Performance Tips

1. **GPU Acceleration**: Ollama supports ROCm (AMD) and CUDA (NVIDIA). Ensure drivers are installed.
2. **Model Selection**: Smaller models (7B) are faster; larger (13B+) are more capable.
3. **Context Length**: Reduce if experiencing slowness; increase for longer conversations.
4. **Memory Usage**: Monitor with `docker stats`

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional tool integrations
- Better memory management
- Enhanced error handling
- Performance optimizations
- Additional search backends

---

## 📝 License

MIT License - Feel free to use this project for personal or commercial purposes.

---

## 🙌 Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM runtime
- [ReAct](https://react-lm.github.io) - Synergizing Reasoning and Acting in LLMs
- [Qwen](https://huggingface.co/Qwen) - Model provider
- Open source community for all the amazing libraries

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section
2. Review Docker logs
3. Ensure Ollama service is running
4. Verify model is properly downloaded

---

**Happy autonomous reasoning! 🚀**
