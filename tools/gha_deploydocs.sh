#!/usr/bin/env bash

set -euo pipefail
set -x

conda config --set channel_priority strict
conda install -y -q heyoka.py numpy scipy ipython sphinx furo nbsphinx pandoc \
    sphinx-copybutton sphinx-design
python -m pip install --no-deps -e .
python -m sphinx -b html doc doc/_build/html