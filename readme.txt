Paper submission source code
============================

This directory contains source code only. It does not contain the NYC dataset,
generated scenarios, routing indexes, SQLite state stores, model checkpoints,
experiment results, figures, or third-party reference datasets.

Directory structure
-------------------
poi_road_sim/          Simulator, queue and waiting-load metrics, routing,
                       MPC/RISE-MPC, SCAIR baseline, PPO policy, training,
                       and experiment runner.
poi_road_sim/native/   C++ source for the H2H routing index and query library.
experiments/           Scenario construction, background generation, regional
                       experiment orchestration, result summaries, and EPS plots.
tests/                 Tests that run without external fixtures or datasets.
*.py                   Top-level expert preparation, BC/DAgger/PPO training,
                       MPC precomputation, and four-scenario evaluation.

Environment
-----------
Run from this directory with Python 3.10 or newer. The simulator and MPC use
the Python standard library. Neural training requires PyTorch and NumPy;
the EPS plot and ablation summary require Matplotlib. Building the native
routing module requires a C++ compiler (g++). Install compatible versions of
these packages in your own environment.

Inputs and execution
--------------------
Supply NYC v4 static road/POI files, prepared scenario JSON files, background
configuration JSON, a routing index and a state store using the command-line
path options of the corresponding scripts. Several legacy commands retain
default paths relative to this directory; those paths refer to external inputs
and are intentionally absent from this source package. Pass explicit paths for
reproduction. Training creates new checkpoints and evaluation creates results
in the output directory specified by the caller.

Key entry points:
  python train_full_ppo.py --dataset DATASET --model-dir MODEL_DIR --output OUTPUT
  python run_four_scenario_eval.py --dataset DATASET --model-dir MODEL_DIR --output OUTPUT
  python experiments/build_scaled_four_scenarios.py --help
  python experiments/summarize_slot_loads.py --help
  python experiments/plot_slot_load_eps.py --input SLOT_LOAD_CSV --output-dir FIGURES

The slot-load plotting script writes separate Road and POI EPS figures for
each scenario, using the first 96 slots by default. For the original NYC
experiments, the prepared scenarios, configuration files, routing index and
trained checkpoint are required in addition to this source package.

Validation
----------
Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests

The original repository also contains tests tied to demonstration JSON files,
legacy experiment scripts, and a native H2H index fixture. Those inputs are
outside the scope of this source-only package, so those tests are omitted.
