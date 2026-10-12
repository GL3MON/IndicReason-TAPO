#!/usr/bin/env python3
"""Compatibility entry point for checkpoint evaluation/export."""
import sys
from indic_reasoning.evaluation import runner as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
