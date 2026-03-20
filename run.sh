#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

clang++ -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude src/*.cpp -o matrix_calculator
./matrix_calculator
