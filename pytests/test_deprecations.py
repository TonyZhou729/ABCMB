import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir + '/../')

import abcmb.main  # importing the module registers its @deprecated functions
import abcmb.constants  # deprecated modules register when imported
import abcmb.ABCMBTools
from abcmb.utils.deprecation import _REGISTRY, _parse
from abcmb.version import __version__


def test_no_overdue_deprecations():
    overdue = [name for name, rm in _REGISTRY if _parse(__version__) >= _parse(rm)]
    assert not overdue, f"Remove before releasing {__version__}: {overdue}"
