import sys
from pathlib import Path

# run from a clone without installing: this folder (cue_livekit) and the repo root (cue)
HERE = Path(__file__).resolve().parents[1]
for path in (HERE, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
