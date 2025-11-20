"""Entry point for running ValtaPyV2 as a module."""

import sys
from .interface.cli import main

if __name__ == "__main__":
    sys.exit(main())