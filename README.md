# Linear Algebra Matrix Calculator

This project is a C++ command-line matrix calculator focused on 12 core matrix operations.

## Build

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude src/*.cpp -o matrix_calculator
```

## Run

```bash
./matrix_calculator
```

## Included operations

- Add two matrices
- Subtract two matrices
- Multiply a matrix by a scalar
- Multiply two matrices
- Transpose a matrix
- Dot product of two vectors
- Row echelon form (REF)
- Reduced row echelon form (RREF)
- Determinant
- Inverse
- Rank
- Solve `Ax = b`

## Project layout

- `include/matrix_solver/types.hpp`: matrix/result data types
- `include/matrix_solver/io.hpp` and `src/io.cpp`: input, formatting, and printing
- `include/matrix_solver/operations.hpp` and `src/operations.cpp`: the 12 supported matrix algorithms
- `include/matrix_solver/cli.hpp` and `src/cli.cpp`: menu flow and command handlers
- `src/main.cpp`: tiny entry point

## Notes

- The calculator uses floating-point arithmetic, so very ill-conditioned matrices may show small numerical error.
- The CLI intentionally stops at the 12 operations listed above.
