#!/usr/bin/env bash

set -euo pipefail
set -x

conda config --set channel_priority strict
conda install -y -q heyoka.py numpy scipy
python -m pip install --no-deps -e .

(
    cd "${HOME}"
    python -c "import quivira; result = quivira.test.run_test_suite(); raise SystemExit(not result.wasSuccessful())"
)