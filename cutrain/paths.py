"""Repository locations. Data, outputs and virtualenvs are found from here, not from each module's folder."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# This repository normally sits inside the assistant-harness checkout; prepare_data and the probes import its src/.
HARNESS_ROOT = REPO.parent
