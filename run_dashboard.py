import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

from app.gradio_dashboard import CSS, THEME, dashboard


if __name__ == "__main__":
    dashboard.queue(default_concurrency_limit=4).launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        theme=THEME,
        css=CSS,
    )
