"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK with 16GB Free RAM)
=============================================================================
Mounts the FastAPI RAG backend + static chat UI alongside Gradio.
"""

import os
import uvicorn
import gradio as gr
from api.app import create_app

# Create the full FastAPI application with all routers & lifespan
fastapi_app = create_app()

# Gradio interface to satisfy Hugging Face Spaces health check & UI preview
with gr.Blocks(title="Multi-Agent RAG Assistant") as demo:
    gr.Markdown("""
    # Multi-Agent RAG Assistant is Live!
    
    - **Web Chat UI**: [Open Chat Interface](/)
    - **API Documentation**: [Interactive Swagger Docs](/docs)
    - **Health Status**: [System Health](/health)
    """)

# Mount Gradio at /gradio so root / serves your clean static ChatGPT UI
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
