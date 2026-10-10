# When to Stop Physics-Informed Learning

An executable study of interval-assisted, anytime-valid error certificates for physics-informed neural networks. The repository includes the complete experimental data, trained network weights, audit observations, and analysis programs under the MIT license.

The primary study contains 80 training trajectories and 400 checkpoint audits across four manufactured PDEs, two tanh architectures, two training samplers, and five random seeds. A separate monitoring experiment contains 20,000 repetitions for each of four Bernoulli means. Eight final networks support a deterministic-refinement comparison.

## Environment

Use Python 3.12 and the pinned packages in `requirements.txt`. The recorded runs used binary64 CPU training and 80-bit Arb residual enclosures. Six single-thread CPU workers ran on an ARM64 operating system with 12 CPU cores and 18 GB RAM. A GPU is optional for exploratory training; the primary results use CPU double precision. A modern six-core CPU and 8–18 GB RAM are suitable starting resources. Reduce `--workers` if memory is constrained. Measured timings depend on hardware and software builds.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_methods.py
python verify_results.py
python analyze.py
```

`verify_results.py` checks the supplied result manifest, all primary configurations, audit ranges, recorded guarantees against independent quadrature, and selected recomputed betting bounds. `analyze.py` regenerates CSV and JSON numerical summaries from the stored audit and training records. This is the fastest way to reproduce the reported tables numerically.

## Regenerating experiments

Run from the repository root. Training and checkpoint auditing reuse existing complete records, so an interrupted run can resume. The distributed `results/` directory holds the reference run. To perform an independent full regeneration while retaining those records, rename it first:

```sh
mv results results-reference
python run_all.py --workers 6
```

The workflow executes:

```sh
python train.py --workers 6 --device cpu --steps 3000 --seed-start 101 --seeds 5
python experiment.py --workers 6 --nmax 8192
python stress_test.py
python refinement.py
python analyze.py
python verify_results.py --numerical-only
```

The original primary study fits comfortably within a day's compute on the stated resource class; the recorded compact-network training is much shorter than the interval audits. Eight deterministic refinements and the repeated-monitoring test are included in the workflow. Numerical results from saved weights should agree to floating-point tolerance; retraining can vary across BLAS, PyTorch, CPU, and operating-system builds. Execution times are measurements, not deterministic outputs.

## Files and data

| File | Purpose |
|---|---|
| `core.py` | Four manufactured PDE definitions, hard boundary/initial constraints, tanh models, automatic and independent analytic differentiation |
| `train.py` | Uniform and residual-mixture training; Adam followed by L-BFGS; saved immutable checkpoints |
| `enclosures.py` | Natural and centered residual enclosures with Arb arithmetic; dyadic cell partitions |
| `certify.py` | Twelve-component bounded-mean betting mixture and conservative bisection inversion |
| `experiment.py` | Fresh checkpoint audits, uniform and interval-guided proposals, corrected Bernstein and Hoeffding comparisons |
| `stress_test.py` | Bernoulli repeated-monitoring experiment, exact binomial intervals, saved count streams |
| `refinement.py` | Uniform deterministic refinement on eight predefined final networks |
| `analyze.py` | Certificate curves, stopping-policy replay, CSV and JSON summaries |
| `run_all.py` | Complete execution sequence |
| `test_methods.py` | Independent differentiation, enclosure, and betting checks |
| `verify_results.py` | Integrity and numerical consistency checks |
| `SHA256SUMS.json` | Hashes of the supplied experimental result files |
| `CITATION.cff` | Machine-readable citation and author metadata |

`results/trajectories/<case>_w<width>d<depth>_<sampler>_s<seed>/` contains `config.json`, `history.json`, and NumPy checkpoint weight archives. Layer `i` has arrays `wi` and `bi`; each matrix follows the PyTorch linear-layer convention (output units by input units). The final label 3200 denotes 3000 Adam updates plus four L-BFGS calls with at most 50 iterations each. It is a nominal budget label, not a count of line-search function evaluations.

`results/audits/<run>/audit_<checkpoint>_m<partition>.json` records cell counts, lower-envelope integral `A`, remaining width `D`, deterministic bounds, quadrature references, confidence bounds, and component times. The matching NPZ stores combined and natural cell bounds and the normalized sample sequences `importance` and `uniform`. Doubling prefixes of these sequences reproduce all nine audit looks. Within each checkpoint the network and proposal remain fixed. Fresh random streams are used at new checkpoints; observations are not pooled across changing networks.

`results/refinement/` records each successive partition, its certificate, and individual and cumulative times. `results/stress/counts.npz` contains Bernoulli cumulative-count arrays with axes (checkpoint, look, repetition); the keys identify the four means. `results/stress/summary.json` records failure counts and exact 95% binomial confidence intervals.

`results/summary/certificate_curves.csv` contains all 14,400 primary bound evaluations. `stopping.csv` and `summary.json` contain stopping outcomes and total costs. The JSON also contains the full-budget ablation. All 80 trajectories remain in cost summaries, including failures to certify.

## Statistical settings

The total error budget is 0.05, with checkpoint allocation `alpha_k = 0.05 / (k * (k + 1))`. Betting fractions are `0.99 * 2**(-ell)` for `ell = 0,...,11`. Audit sizes are 32 through 8192 by doubling. Fixed-time Bernstein and Hoeffding bounds additionally use `delta_kj = alpha_k / (j * (j + 1))` at look `j`. The target is absolute L2 solution error, using the PDE-specific stability constants defined in the study.

The economical stopping policy first accepts a sufficient deterministic bound. It skips sampling if the integrated lower envelope already exceeds the residual threshold. Otherwise, it checks doubling prefixes, accepts only a valid upper certificate, and defers an unsuccessful audit if its observed residual estimate exceeds the threshold. Deferral can withhold a certificate but cannot create an acceptance. The full-budget ablation instead exhausts each unsuccessful audit. These decisions are replayed on saved trajectories, with measured component times summed; they are not separately timed closed-loop training runs.

The theoretical probability statement assumes valid cell enclosures and the specified conditional sampling law. Arb encloses the residual calculations, while scalar statistical reductions use binary64 arithmetic. The implementation and numerical checks are not an end-to-end formal verification of every floating-point operation.

## Data provenance and license

All PDE data are generated analytically from the four specified manufactured solutions. The optimizer receives forcing values and constraints, while exact solutions support independent error assessment. Training seeds are 101–105; the audit seed formula is stored in `experiment.py`; the repeated-monitoring seed is 78213. The software and generated experimental data are distributed under the MIT license in `LICENSE`.
