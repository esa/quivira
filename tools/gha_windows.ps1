$ErrorActionPreference = "Stop"

conda config --set channel_priority strict
conda install -y heyoka.py numpy scipy
python -m pip install --no-deps -e .

Push-Location $env:TEMP
try {
    python -c "import quivira; result = quivira.test.run_test_suite(); raise SystemExit(not result.wasSuccessful())"
}
finally {
    Pop-Location
}