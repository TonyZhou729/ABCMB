#!/usr/bin/env bash
#
# Run the linter, formatter check, and tests in one step (mirrors CI).
#
#   ./check.sh          lint + format-check + tests (check-only, like CI)
#   ./check.sh fix      auto-apply ruff format + fixable lint, then test
#   ./check.sh test     run the tests only (skip lint/format)
#
set -euo pipefail

RUFF="uvx ruff@0.15.20"

usage() {
    cat <<'EOF'
Run the linter, formatter check, and tests (mirrors CI).

Usage: ./check.sh [command]

Commands:
  (none)   lint + format-check + tests (check-only, like CI)
  fix      auto-apply ruff format + fixable lint, then run tests
  test     run the tests only (skip lint/format)
  help     show this help
EOF
}

case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    test)
        ;;  # skip ruff, just run tests below
    fix)
        echo ">> ruff format (applying)"
        $RUFF format .
        echo ">> ruff check --fix (applying)"
        $RUFF check --fix .
        ;;
    "")
        echo ">> ruff check"
        $RUFF check .
        echo ">> ruff format --check"
        $RUFF format --check .
        ;;
    *)
        echo "error: unknown command '$1'" >&2
        usage >&2
        exit 2
        ;;
esac

echo ">> pytest"
uv run --extra test pytest -s -vv pytests
