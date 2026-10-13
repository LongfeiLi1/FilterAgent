"""Local paths configured without private files or credentials."""
import os
from pathlib import Path
CODE_ROOT=Path(__file__).resolve().parents[2]
BENCHMARK_ROOT=Path(os.environ.get('FILTERAGENT_BENCHMARK',str(CODE_ROOT.parent/'FilterAgent-benchmark'))).expanduser().resolve()
BENCHMARK_DIR=BENCHMARK_ROOT/'data/filteragent_v3'
PDK=Path(os.environ.get('FILTERAGENT_PDK_LIB',str(CODE_ROOT/'external/sky130/libs.tech/ngspice/sky130.lib.spice'))).expanduser().resolve()
