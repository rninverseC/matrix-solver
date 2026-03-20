# Linear Algebra Matrix Calculator

This project is a C++ command-line matrix calculator aimed at the main computational topics from Lay's linear algebra book.

## Build

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude src/*.cpp -o matrix_calculator
```

## Run

```bash
./matrix_calculator
```

## Included operations

- Matrix addition, subtraction, scalar multiplication, transpose, and matrix multiplication
- Dot products
- REF and RREF
- Determinant, inverse, and rank
- Solving `Ax = b`
- Bases for the row space, column space, and null space
- PLU factorization
- Reduced QR factorization with Gram-Schmidt
- Orthogonal projection and least-squares solutions
- Characteristic polynomial
- Dominant eigenpair via the power method
- Symmetric eigendecomposition
- Compact SVD
- Moore-Penrose pseudoinverse

## Project layout

- `include/matrix_solver/types.hpp`: matrix/result data types
- `include/matrix_solver/io.hpp` and `src/io.cpp`: input, formatting, and printing
- `include/matrix_solver/operations.hpp` and `src/operations.cpp`: matrix algorithms and decompositions
- `include/matrix_solver/cli.hpp` and `src/cli.cpp`: menu flow and command handlers
- `src/main.cpp`: tiny entry point

## Notes

- The calculator uses floating-point arithmetic, so very ill-conditioned matrices may show small numerical error.
- The spectral tools are split into:
  - a general dominant-eigenvalue approximation via the power method
  - a full eigendecomposition routine for symmetric matrices
