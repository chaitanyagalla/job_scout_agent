"""FastAPI server entry point for local dev and deployment."""
import os

from google.adk.cli.fast_api import get_fast_api_app

try:
    import google.cloud.logging
    google.cloud.logging.Client().setup_logging()
except Exception:
    pass

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=os.getenv("SESSION_SERVICE_URI"),
    artifact_service_uri=os.getenv("ARTIFACT_SERVICE_URI"),
    allow_origins=["*"],
    web=False,
    use_local_storage=True,
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)