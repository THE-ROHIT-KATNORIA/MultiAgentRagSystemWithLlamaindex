"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK with ZeroGPU / CPU)
=============================================================================
Mounts the FastAPI RAG backend + static chat UI alongside Gradio.
"""

import os
# Disable Gradio 5 Server-Side Rendering (SSR) Node proxy to prevent port 7861 binding conflicts
os.environ["GRADIO_SSR_MODE"] = "false"

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            return lambda f: f

import uvicorn
import gradio as gr
from api.app import create_app

# Create the full FastAPI application with all routers & lifespan
fastapi_app = create_app()

@spaces.GPU
def _zero_gpu_worker():
    """Satisfies Hugging Face ZeroGPU startup validation."""
    pass

def query_agent(prompt: str) -> str:
    """In-process query handler calling the local RAG pipeline directly."""
    if not prompt or not prompt.strip():
        return "Please enter a question."
    import httpx
    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.post(
                "http://127.0.0.1:7860/rag/query",
                json={"question": prompt, "session_id": "gradio_session"}
            )
            if res.status_code == 200:
                return res.json().get("answer", "No response received.")
            return f"Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"Error: {e}"

# Gradio interface to satisfy Hugging Face Spaces root health check & UI preview
with gr.Blocks(title="Multi-Agent RAG Assistant") as demo:
    gr.Markdown("""
    # Multi-Agent RAG Assistant
    
    Enterprise Multi-Agent RAG with LlamaIndex, PostgreSQL/pgvector, and Groq Cloud.
    
    - **Clean Web Chat UI**: [Open Chat Interface](/chat)
    - **Interactive API Documentation**: [Swagger Docs](/docs)
    - **System Health Status**: [Health Status](/health)
    """)
    with gr.Row():
        text_in = gr.Textbox(placeholder="Ask anything...", label="Prompt", lines=2)
    text_out = gr.Markdown(label="Response")
    btn = gr.Button("Submit", variant="primary")
    btn.click(fn=query_agent, inputs=text_in, outputs=text_out)

# Mount Gradio at root "/" so Hugging Face Spaces health check passes
app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    try:
        from spaces import zero as _spaces_zero
        if hasattr(_spaces_zero, "startup"):
            _spaces_zero.startup()
            print("[app] ZeroGPU startup report sent successfully", flush=True)
        else:
            print("[app] spaces.zero.startup not present (running off ZeroGPU)", flush=True)
    except Exception as error:
        print(f"[app] ZeroGPU startup skipped/failed: {error!r}", flush=True)

    port = int(os.getenv("PORT", "7860"))
    if port != 7860:
        port = 7860
    uvicorn.run(app, host="0.0.0.0", port=port)

