"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK with 16GB Free RAM)
=============================================================================
Mounts the FastAPI RAG backend + static chat UI alongside Gradio.
"""

import os
# Disable Gradio 5 Server-Side Rendering (SSR) Node proxy to prevent port 7861 binding conflicts
os.environ["GRADIO_SSR_MODE"] = "false"

import uvicorn
import gradio as gr
from api.app import create_app

# Create the full FastAPI application with all routers & lifespan
fastapi_app = create_app()

# Gradio interface to satisfy Hugging Face Spaces root health check & UI preview
with gr.Blocks(title="Multi-Agent RAG Assistant") as demo:
    gr.Markdown("""
    # Multi-Agent RAG Assistant
    
    Enterprise Multi-Agent RAG with LlamaIndex, PostgreSQL/pgvector, and Groq Cloud.
    
    - **Clean Web Chat UI**: [Open Chat Interface](/chat)
    - **Interactive API Documentation**: [Swagger Docs](/docs)
    - **System Health Status**: [Health Status](/health)
    """)

# Mount Gradio at root "/" so Hugging Face Spaces health check passes
app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    if port != 7860:
        port = 7860
    uvicorn.run(app, host="0.0.0.0", port=port)

