"""FastAPI server entry point for local development."""
import os

from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.agent_loader import AgentLoader

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "job_scout"


class JobScoutAgentLoader(AgentLoader):
    """Limit ADK app discovery to the production Job Scout agent."""

    def list_agents(self) -> list[str]:
        return [APP_NAME]


app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    agent_loader=JobScoutAgentLoader(AGENT_DIR),
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
