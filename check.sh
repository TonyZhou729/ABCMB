#!/usr/bin/env bash
#
# Run the linter and the tests in one step (mirrors CI).
#
#   ./check.sh          lint + tests
#   ./check.sh fix      auto-apply fixable lint, then test
#   ./check.sh test     run the tests only (skip lint)
#
# Setup:
#   pip install -r requirements-dev.txt      # ruff, pre-commit
#   pip install -r pytest_requirements.txt   # test dependencies
#   pip install -e . --no-deps               # make `abcmb` importable
#
# NOTE: the ruff *formatter* is configured but not switched on — see the
# commented-out lines below and in .pre-commit-config.yaml.
#
set -euo pipefail

usage() {
    cat <<'USAGE'
Run the linter and the tests (mirrors CI).

Usage: ./check.sh [command]

Commands:
  (none)   lint + tests
  fix      auto-apply fixable lint, then run tests
  test     run the tests only (skip lint)
  help     show this help
USAGE
}

case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    test)
        ;;  # skip ruff, just run tests below
    fix)
        # Uncomment once the one-off `ruff format .` sweep has landed:
        # echo ">> ruff format (applying)"
        # ruff format .
        echo ">> ruff check --fix (applying)"
        ruff check --fix .
        ;;
    "")
        echo ">> ruff check"
        ruff check .
        # Uncomment once the one-off `ruff format .` sweep has landed:
        # echo ">> ruff format --check"
        # ruff format --check .
        ;;
    *)
        echo "error: unknown command '$1'" >&2
        usage >&2
        exit 2
        ;;
esac

echo ">> pytest"
pytest -s -vv pytests
