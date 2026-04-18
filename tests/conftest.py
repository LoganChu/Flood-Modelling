import sys
from pathlib import Path

_EXTS_ROOT = Path(__file__).resolve().parent.parent / "exts"
if str(_EXTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTS_ROOT))
