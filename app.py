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

# Gradio interface to satisfy Hugging Face Spaces health check & UI preview
with gr.Blocks(title="Multi-Agent RAG Assistant") as demo:
    gr.Markdown("""
    # Multi-Agent RAG Assistant is Live!
    
    - **Web Chat UI**: [Open Chat Interface](/)
    - **API Documentation**: [Interactive Swagger Docs](/docs)
    - **Health Status**: [System Health](/health)
    """)

# Mount Gradio with ssr_mode=False so it only uses pure FastAPI client-side routes
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio", ssr_mode=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    if port != 7860:
        port = 7860
    uvicorn.run(app, host="0.0.0.0", port=port)

