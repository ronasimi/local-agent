import config

def sub_agent_task(text: str, instruction: str) -> str:
    """Passes messy or token-heavy text to the 0.5B model for extraction/summarization."""
    if not text or "error" in text.lower() or len(text) < 200: 
        return text 
    try:
        res = config.client.chat(model=config.SUB_MODEL_NAME, messages=[
            {"role": "system", "content": f"You are a concise data-extraction sub-agent. {instruction} No fluff."},
            {"role": "user", "content": text[:8000]}
        ], options={"num_predict": 300, "temperature": 0.1})
        return f"[0.5B Summary] {res.message.content.strip()}"
    except Exception as e: 
        return f"[Sub-agent error: {e}]\n<raw_text>\n{text[:500]}\n</raw_text>"
