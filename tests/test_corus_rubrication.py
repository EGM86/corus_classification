import sys
sys.path.append('../corus_rubrication')

from corus_rubrication import __version__

def test_version():
    assert __version__ == '0.1.0'
