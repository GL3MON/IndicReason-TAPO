#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in indic_reasoning."""
import sys
from indic_reasoning.training import rl as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
