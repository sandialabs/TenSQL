#!/usr/bin/env python3
"""Custom test harness for tensql"""

import os
import unittest
import coverage
import pathlib
import warnings

def main():
    #warnings.simplefilter('always', DeprecationWarning)
    testsdir = (pathlib.Path(__file__).parent / 'tensql' / 'test').resolve()
    packagedir = testsdir.parent

    covdir = pathlib.Path("./coverage").resolve()
    covdir.mkdir(exist_ok=True)

    covhtml = covdir / "html"
    covhtml.mkdir(exist_ok=True)

    covdata = covdir / "datafile"

    cov = coverage.Coverage(
        data_file = str(covdata.resolve()),
        include = str((packagedir / "*").resolve()),
        omit = [
            str((testsdir / "*").resolve()),
            str((packagedir / "_version.py").resolve())
        ],
        config_file = str((testsdir / "coveragerc").resolve())
    )

    cov.start()
    suite = unittest.TestSuite()
    import tensql
    from tensql.test.test_util import test_registry

    for test in test_registry['all']:
        suite.addTest(test())

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)
    cov.stop()
    cov.save()
    cov.html_report(directory=str(covhtml))

main()

