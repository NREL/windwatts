#!/bin/bash

# Navigate to API directory
cd windwatts-api || exit 1

# Try to activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Capture arguments (files)
FILES="$@"
if [ -z "$FILES" ]; then
    FILES="."
fi

echo "Running Python Lint checks (Ruff)..."
if ! ruff check $FILES; then
    echo "" >&2
    echo "--------------------------------------------------------" >&2
    echo "❌ Python linting failed!" >&2
    echo "   Please run the following to fix issues:" >&2
    echo "   make lint    (or 'make -C windwatts-api lint')" >&2
    echo "--------------------------------------------------------" >&2
    exit 1
fi

echo "Running Python Format checks (Ruff)..."
if ! ruff format --check $FILES; then
    echo "" >&2
    echo "--------------------------------------------------------" >&2
    echo "❌ Python formatting failed!" >&2
    echo "   Please run the following to fix issues:" >&2
    echo "   make format  (or 'make -C windwatts-api format')" >&2
    echo "--------------------------------------------------------" >&2
    exit 1
fi

echo "✅ Python Lint and Format checks passed."
exit 0
