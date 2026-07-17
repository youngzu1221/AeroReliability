from pathlib import Path
from runpy import run_path
import sys


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    run_path(str(project_root / "app.py"), run_name="__main__")
