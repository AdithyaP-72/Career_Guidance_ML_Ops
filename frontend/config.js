// Where the API lives. This committed version is for local development
// (run the API with: uv run uvicorn src.serving.app:app). On Render, the static site's build
// command overwrites this file with the deployed API's URL (see render.yaml).
window.API_URL = "http://localhost:8000";
