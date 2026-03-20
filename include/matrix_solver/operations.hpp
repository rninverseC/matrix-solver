#pragma once

#include "matrix_solver/types.hpp"

#include <string>
#include <vector>

namespace matrix_solver {

std::vector<ld> getColumn(const Matrix& matrix, int column);
void setColumn(Matrix& matrix, int column, const std::vector<ld>& values);
Matrix fromColumns(const std::vector<std::vector<ld>>& columns, int rowCount);

ld dotProduct(const std::vector<ld>& a, const std::vector<ld>& b);
ld vectorNorm(const std::vector<ld>& values);
std::vector<ld> scaleVector(const std::vector<ld>& values, ld scalar);
std::vector<ld> addVectors(const std::vector<ld>& a, const std::vector<ld>& b);
std::vector<ld> subtractVectors(const std::vector<ld>& a, const std::vector<ld>& b);

Matrix addMatrices(const Matrix& a, const Matrix& b);
Matrix subtractMatrices(const Matrix& a, const Matrix& b);
Matrix scalarMultiply(const Matrix& matrix, ld scalar);
Matrix transpose(const Matrix& matrix);
Matrix multiplyMatrices(const Matrix& a, const Matrix& b);
std::vector<ld> multiplyMatrixVector(const Matrix& matrix, const std::vector<ld>& vector);
Matrix augmentMatrices(const Matrix& left, const Matrix& right);

EliminationResult toRef(Matrix matrix);
EliminationResult toRref(Matrix matrix);
int rankOfMatrix(const Matrix& matrix);
ld determinant(const Matrix& matrix);
Matrix inverseMatrix(const Matrix& matrix);

std::vector<std::vector<ld>> rowSpaceBasis(const Matrix& matrix);
std::vector<std::vector<ld>> columnSpaceBasis(const Matrix& matrix);
std::vector<std::vector<ld>> nullSpaceBasis(const Matrix& matrix);
LinearSystemResult solveLinearSystem(const Matrix& A, const std::vector<ld>& b);

LUResult pluDecomposition(const Matrix& matrix);
QRResult reducedQR(const Matrix& matrix);
std::vector<ld> projectOntoColumnSpace(const Matrix& A, const std::vector<ld>& b);

bool isSymmetric(const Matrix& matrix, ld tolerance = 1e-8L);
SymmetricEigenResult jacobiEigenDecomposition(const Matrix& matrix);
SVDResult compactSVD(const Matrix& matrix);
Matrix pseudoInverse(const Matrix& matrix);
std::vector<ld> leastSquaresSolution(
    const Matrix& A,
    const std::vector<ld>& b,
    std::vector<ld>* projection = nullptr,
    std::vector<ld>* residual = nullptr);

std::vector<ld> characteristicPolynomialCoefficients(const Matrix& matrix);
std::string characteristicPolynomialToString(const std::vector<ld>& coefficients);
PowerMethodResult dominantEigenpair(const Matrix& matrix, int maxIterations = 1000, ld tolerance = 1e-10L);

}  // namespace matrix_solver
