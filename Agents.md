# Code Modification Guidelines

This tool is used to extract data from the GRID and build training datasets. Some guidelines:

1. If you modify the cli, make sure to update `README.md`
1. Tests should be in a `test_xxx.py` file where `xxx` is the name of the file that contains the thing you are testing.
1. Tests should be run with pytest. When developing a test run just that tests, but then run all tests when done just in case something broke. Make sure to use `python -m pytest` with `python` from the appropriate venv (most often in `.venv` in the root directory).
