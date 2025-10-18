# COMPASS — A Compact, Trapdoor-Free Lattice Accumulator

This repository contains a reference implementation of **COMPASS**, a PASS-lineage,
trapdoor-free lattice accumulator with constant-size membership witnesses and
constant-time verification. It accompanies the paper:

> *COMPASS: Compact PASS-Lineage Accumulators with Quantitative Analysis and Implementation* (2025).

## Features

- Transparent setup (no trapdoors)
- PASS_G-style rejection sampling and domain-separated challenges
- Constant-size witnesses; constant-time verification
- Reproducible benchmarks for two PASS_G-style parameter sets

## Repository layout

```
COMPASS/
├─ compass/
│  ├─ __init__.py          # package export
│  ├─ compass.py           # COMPASS class (algorithms)
│  └─ utils.py             # math utils, PartialFourier, packing, hashing
├─ notebooks/
│  └─ COMPASS_release_version.ipynb      # Google Colab demo file
├─ tests/
│  └─ test_compass.py      # end-to-end test / benchmark
├─ requirements.txt        # numpy only for now
├─ LICENSE
└─ README.md
```

## Requirements

- Python 3.9+ (tested on macOS and Colab)
- `numpy`

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (minimal):

```
numpy>=1.23
```

## Quick start

Run the end-to-end test/benchmark:

```bash
python -m tests.test_compass
```

Expected output (abridged, will vary with RNG):

```
TESTING CORRECTED COMPASS
...
COMPASS initialized with set1: N=512, q=205207553, κ=44, t=256
...
ACCUMULATE (K=200 members)
  Member 1: accepted after ...
...
Witness 0: 4288 bytes
...
Average verify time: ~0.8 ms
ALL TESTS PASSED! ✓
```

Switch parameter sets by editing `param_set` in `tests/test_compass.py`:
- `set1`: N=512, q=205,207,553, κ=44, σ=11,336, t=256
- `set2`: N=1024, q=4,294,957,057, κ=36, σ=167,771, t=512

## Interactive notebook

A ready-to-run Google Colab/Jupyter notebook is included under:

```
notebooks/COMPASS_release_version.ipynb
```

This notebook reproduces the results and tables reported in the paper.  
It executes the full COMPASS workflow — **Setup → Accumulate → Witness → Verify** —  
and prints witness sizes, acceptance rates, and verification times for the published parameter sets.

To try it:
- **In Colab:** simply upload the repository and open the notebook; all dependencies install automatically.  
- **Locally:** open the notebook with Jupyter in the activated virtual environment (`.venv`) and run all cells.

The notebook is self-contained and uses the same codebase as the `tests/test_compass.py` benchmark,  
so reviewers can reproduce our measurements without additional configuration.

## Reproducing paper numbers

We report (per paper) witness sizes and timings using the same packing and
partial-Fourier backend as this code. The test prints:
- empirical acceptance rate (attempts per member)
- witness size (serialized bytes)
- verification time (per witness)
- accumulator digest size

## License

Released under the MIT License (see `LICENSE`).

## Acknowledgments

This code draws on the PASS lineage (PASS_RS, PASS_G) for challenge formats
and rejection-sampling ideas. Any mistakes are ours.
