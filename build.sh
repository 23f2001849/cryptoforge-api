#!/bin/bash
# CryptoForge — Render Build Script
# This runs during Render's build phase.
# It installs Python deps AND compiles the C library.

set -e

echo "=== CryptoForge Build ==="

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Compile C library for Linux
echo "Compiling matrixhash.c → matrixhash.so ..."
if [ -f "matrixhash.c" ]; then
    gcc -O2 -shared -fPIC -o matrixhash.so matrixhash.c
    echo "✓ matrixhash.so compiled ($(stat -c%s matrixhash.so 2>/dev/null || echo '?') bytes)"
else
    echo "⚠ matrixhash.c not found — C fast-path will be unavailable"
fi

echo "=== Build complete ==="
