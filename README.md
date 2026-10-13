# FilterAgent

FilterAgent sizes active-RC filters using circuit measurements and a Qwen-based decision loop. It supports Sallen–Key low-pass, Sallen–Key high-pass, and multiple-feedback low-pass circuits, and saves sizing decisions, measurement evidence, and final results for each case.

## Install

Use Python 3.10 or later. The reference dependency set uses Python 3.10.18, NumPy 1.26.4, and SciPy 1.15.3.

```sh
git clone https://github.com/LongfeiLi1/FilterAgent.git
cd FilterAgent
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-reference.txt
```

For other supported Python versions, use `requirements.txt`. Run the commands below from the repository root; the resource files must stay with the source checkout.

## First run

The benchmark data is supplied separately. Before validating or running a case, point `--benchmark` at a compatible `FilterAgent-benchmark` directory containing `MANIFEST.json` and `data/filteragent_v3/`. The CLI expects the complete 45-case suite, even when selecting one case. The default location is `../FilterAgent-benchmark`.

Validate the data and circuit bindings first:

```sh
python run.py --case V3-T1-L1-C01 --benchmark ../FilterAgent-benchmark --check-only
```

This checks the source and data without starting ngspice or making model requests. To run the same case after configuring the simulator, PDK, and API access below:

```sh
python run.py --case V3-T1-L1-C01 --benchmark ../FilterAgent-benchmark --execute --out output/example
```

Use a fresh output directory for every run. Per-case results are saved under `output/example/<case-id>/`, including `result.json` on a completed run or `entry_failure.json` on an execution error. `output/example/summary.json` records success or failure for each selected case. Output directories must be inside this checkout's `output/` directory.

## Configure simulation and model access

Install [ngspice 46](https://ngspice.sourceforge.io/download.html) with KLU and support for the `mdump` command. FilterAgent uses matrix dumps for local small-signal stability checks. Use `--ngspice` if the executable is not on your PATH.

Install SKY130 separately using [open_pdks](https://github.com/RTimothyEdwards/open_pdks) and the [SkyWater PDK sources](https://github.com/google/skywater-pdk). Point `FILTERAGENT_PDK_LIB` at the installed `libs.tech/ngspice/sky130.lib.spice` file. Model files and simulator binaries are not included.

```sh
export FILTERAGENT_PDK_LIB="/path/to/sky130/libs.tech/ngspice/sky130.lib.spice"
export FILTERAGENT_MODEL="qwen3.8-flash"
export FILTERAGENT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

Set `FILTERAGENT_API_KEY` in your shell session using your usual secret manager. The adapter accepts the Aliyun HTTPS endpoint above, and the decision transport requires the recorded `qwen3.8-flash` model ID. Your account must have access to that model. `config.example.json` lists the environment variables; the program does not load that file.

The original PDK commit was not recorded. Relative model-file hashes and the simulator hash are retained in `src/filteragent_r7a/pdk_manifest.json` for comparison. A different simulator build or PDK model set can change the results.

## Batch runs and tests

```sh
python run.py --all --benchmark ../FilterAgent-benchmark --check-only
python run.py --all --benchmark ../FilterAgent-benchmark --execute --out output/batch
python run.py --help
```

`--all` processes the suite sequentially. A failed case is recorded and the remaining cases continue; the command exits nonzero if any case fails. `--execute` enables physical simulation and paid model requests when a model decision is needed. Validation alone does not establish physical success.

For the offline test suite:

```sh
python -m pip install -r requirements-test.txt
python -m pytest
```

Tests use synthetic fixtures and block network calls and simulator launches.

## License

See [LICENSE](LICENSE) for the existing MIT license and [NOTICE.md](NOTICE.md) for the adapted OTA's attribution and dependency notices.
