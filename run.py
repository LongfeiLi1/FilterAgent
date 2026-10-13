"""Run the public Full interface from this checkout."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'src'),str(root),str(root/'benchmark_interfaces'),str(root/'orchestration/filteragent_v3')]
from filteragent_release.cli import main
if __name__=='__main__':raise SystemExit(main())
