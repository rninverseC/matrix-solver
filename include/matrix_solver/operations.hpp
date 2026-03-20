#pragma once

#include "matrix_solver/types.hpp"

#include <vector>

namespace matrix_solver {

ld dotProduct(const std::vector<ld>& a, const std::vector<ld>& b);

Matrix addMatrices(const Matrix& a, const Matrix& b);
Matrix subtractMatrices(const Matrix& a, const Matrix& b);
Matrix scalarMultiply(const Matrix& matrix, ld scalar);
Matrix transpose(const Matrix& matrix);
Matrix multiplyMatrices(const Matrix& a, const Matrix& b);

EliminationResult toRef(Matrix matrix);
EliminationResult toRref(Matrix matrix);
int rankOfMatrix(const Matrix& matrix);
ld determinant(const Matrix& matrix);
Matrix inverseMatrix(const Matrix& matrix);
LinearSystemResult solveLinearSystem(const Matrix& A, const std::vector<ld>& b);

}  // namespace matrix_solver
