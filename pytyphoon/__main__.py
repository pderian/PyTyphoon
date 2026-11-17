"""Allow running pytyphoon as a module.

This enables: python -m pytyphoon --demo particles
Instead of: python -m pytyphoon.cli --demo particles
"""
from .cli import main

if __name__ == "__main__":
    main()