"""Test runner for the quivira package.

Test cases live in separate modules named ``test_*.py`` inside the package.
This keeps the runner stable while new functionality can add focused test
modules alongside the implementation.
"""

import os
import unittest as _ut


def run_test_suite(verbosity=2):
    """Discover and run tests located in ``quivira/test_*.py``.

    Args:
        verbosity (:class:`int`, optional): Verbosity passed to the test
            runner. Default is 2.

    Returns:
        :class:`unittest.TestResult`: The result returned by the test runner.
    """
    import quivira as _qv

    tests_dir = _qv.__path__[0]
    package_root = os.path.dirname(tests_dir)
    suite = _ut.defaultTestLoader.discover(
        start_dir=tests_dir, pattern="test_*.py", top_level_dir=package_root
    )
    runner = _ut.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)
