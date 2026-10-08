import sys
from pathlib import Path

# The research package lives in src/ and is not necessarily pip-installed
# (see README "Setup"), so make it importable from the app the same way
# scripts/ does.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
