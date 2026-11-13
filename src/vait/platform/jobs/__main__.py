"""Run the platform worker command-line entry point."""

from vait.platform.jobs.bootstrap import main

if __name__ == "__main__":
    raise SystemExit(main())
