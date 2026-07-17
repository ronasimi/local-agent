import os
import base64
import config
from .subagent import sub_agent_task

def read_pdf(filename: str) -> str:
    """Extract text from a standard PDF file located in the workspace."""
    try:
        import fitz
    except ImportError:
        return "Error: PyMuPDF (fitz) is not installed. Add 'pymupdf' to requirements.txt."
        
    path = os.path.join(config.WORKSPACE_DIR, os.path.basename(filename))
    if not os.path.exists(path): return f"File '{filename}' not found."
    
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        if len(text) > 4000:
            return sub_agent_task(text, "Extract and summarize the core information from this PDF document.")
        return text[:4000]
    except Exception as e:
        return f"PDF Error: {e}"

def analyze_image(filename: str, prompt: str = "Describe this image in detail.") -> str:
    """Use this tool to evaluate an image or a scanned document. 
    Pass the file name and a prompt asking what you want to know about the image."""
    path = os.path.join(config.WORKSPACE_DIR, os.path.basename(filename))
    if not os.path.exists(path): return f"Error: Image '{filename}' not found in workspace."
    
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        res = config.client.chat(model=config.VISION_MODEL_NAME, messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [encoded_string]
            }
        ], options={"temperature": 0.1})
        
        return f"[Vision Sub-Agent] {res.message.content.strip()}"
    except Exception as e:
        return f"Vision Error: {e}"
