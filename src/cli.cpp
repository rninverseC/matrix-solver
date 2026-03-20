#include "matrix_solver/cli.hpp"

#include "matrix_solver/io.hpp"
#include "matrix_solver/operations.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace matrix_solver {

namespace {

std::string pivotColumnList(const std::vector<int>& pivotColumns) {
    if (pivotColumns.empty()) {
        return "none";
    }
    std::ostringstream out;
    for (std::size_t i = 0; i < pivotColumns.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << (pivotColumns[i] + 1);
    }
    return out.str();
}

void handleAddition() {
    const Matrix A = readMatrix("A");
    const Matrix B = readMatrix("B");
    printMatrix(addMatrices(A, B), "A + B");
}

void handleSubtraction() {
    const Matrix A = readMatrix("A");
    const Matrix B = readMatrix("B");
    printMatrix(subtractMatrices(A, B), "A - B");
}

void handleScalarMultiplication() {
    const Matrix A = readMatrix("A");
    const ld scalar = readLongDouble("Scalar: ");
    printMatrix(scalarMultiply(A, scalar), formatNumber(scalar) + " * A");
}

void handleMatrixMultiplication() {
    const Matrix A = readMatrix("A");
    const Matrix B = readMatrix("B");
    printMatrix(multiplyMatrices(A, B), "A * B");
}

void handleTranspose() {
    const Matrix A = readMatrix("A");
    printMatrix(transpose(A), "A^T");
}

void handleDotProduct() {
    const int length = readPositiveInt("Vector length: ");
    const auto u = readVector("u", length);
    const auto v = readVector("v", length);
    std::cout << "u . v = " << formatNumber(dotProduct(u, v)) << "\n";
}

void handleRef() {
    const Matrix A = readMatrix("A");
    const auto result = toRef(A);
    printMatrix(result.matrix, "REF(A)");
    std::cout << "Rank = " << result.rank << "\n";
    std::cout << "Pivot columns (1-based) = " << pivotColumnList(result.pivotColumns) << "\n";
}

void handleRref() {
    const Matrix A = readMatrix("A");
    const auto result = toRref(A);
    printMatrix(result.matrix, "RREF(A)");
    std::cout << "Rank = " << result.rank << "\n";
    std::cout << "Pivot columns (1-based) = " << pivotColumnList(result.pivotColumns) << "\n";
}

void handleDeterminant() {
    const Matrix A = readMatrix("A");
    std::cout << "det(A) = " << formatNumber(determinant(A)) << "\n";
}

void handleInverse() {
    const Matrix A = readMatrix("A");
    printMatrix(inverseMatrix(A), "A^(-1)");
}

void handleRank() {
    const Matrix A = readMatrix("A");
    std::cout << "rank(A) = " << rankOfMatrix(A) << "\n";
}

void handleSolveLinearSystem() {
    const Matrix A = readMatrix("A");
    const auto b = readVector("b", A.rows);
    const auto solution = solveLinearSystem(A, b);

    printMatrix(solution.augmentedRref, "RREF([A|b])");
    if (!solution.consistent) {
        std::cout << "The system is inconsistent, so it has no solution.\n";
        return;
    }

    if (solution.unique) {
        printVector(solution.particularSolution, "Unique solution x");
        return;
    }

    printVector(solution.particularSolution, "One particular solution x_p");
    printBasis(solution.nullSpaceBasis, "Null space");
    std::cout << "General solution: x = x_p";
    for (std::size_t i = 0; i < solution.nullSpaceBasis.size(); ++i) {
        std::cout << " + c" << (i + 1) << "*v" << (i + 1);
    }
    std::cout << "\n";
}

void handleSpaces() {
    const Matrix A = readMatrix("A");
    printBasis(rowSpaceBasis(A), "Row space");
    printBasis(columnSpaceBasis(A), "Column space");
    printBasis(nullSpaceBasis(A), "Null space");
}

void handlePlu() {
    const Matrix A = readMatrix("A");
    const auto result = pluDecomposition(A);
    if (!result.success) {
        std::cout << result.message << "\n";
        return;
    }
    std::cout << result.message << "\n";
    printMatrix(result.P, "P");
    printMatrix(result.L, "L");
    printMatrix(result.U, "U");
}

void handleQr() {
    const Matrix A = readMatrix("A");
    const auto result = reducedQR(A);
    std::cout << result.message << "\n";
    std::cout << "Rank = " << result.rank << "\n";
    printMatrix(result.Q, "Q");
    printMatrix(result.R, "R");
}

void handleProjection() {
    const Matrix A = readMatrix("A");
    const auto b = readVector("b", A.rows);
    printVector(projectOntoColumnSpace(A, b), "proj_Col(A)(b)");
}

void handleLeastSquares() {
    const Matrix A = readMatrix("A");
    const auto b = readVector("b", A.rows);
    std::vector<ld> projection;
    std::vector<ld> residual;
    const auto x = leastSquaresSolution(A, b, &projection, &residual);
    printVector(x, "Least-squares solution x_hat");
    printVector(projection, "Projection A*x_hat");
    printVector(residual, "Residual b - A*x_hat");
    std::cout << "Residual norm = " << formatNumber(vectorNorm(residual)) << "\n";
}

void handleCharacteristicPolynomial() {
    const Matrix A = readMatrix("A");
    const auto coefficients = characteristicPolynomialCoefficients(A);
    std::cout << "Characteristic polynomial:\n";
    std::cout << "  " << characteristicPolynomialToString(coefficients) << "\n";
    std::cout << "Coefficients [1, c1, c2, ..., cn] = " << formatVector(coefficients) << "\n";
}

void handlePowerMethod() {
    const Matrix A = readMatrix("A");
    const auto result = dominantEigenpair(A);
    if (!result.success) {
        std::cout << result.message << "\n";
        return;
    }
    std::cout << result.message << "\n";
    std::cout << "Iterations = " << result.iterations << "\n";
    std::cout << "Approximate dominant eigenvalue = " << formatNumber(result.eigenvalue) << "\n";
    std::cout << "Residual norm = " << formatNumber(result.residualNorm) << "\n";
    printVector(result.eigenvector, "Approximate dominant eigenvector");
}

void handleSymmetricEigen() {
    const Matrix A = readMatrix("A");
    const auto result = jacobiEigenDecomposition(A);
    if (!result.success) {
        std::cout << result.message << "\n";
        return;
    }
    std::cout << result.message << "\n";
    std::cout << "Eigenvalues = " << formatVector(result.eigenvalues) << "\n";
    printMatrix(result.eigenvectors, "Eigenvector matrix Q (columns are eigenvectors)");
}

void handleSvd() {
    const Matrix A = readMatrix("A");
    const auto result = compactSVD(A);
    if (!result.success) {
        std::cout << result.message << "\n";
        return;
    }
    std::cout << result.message << "\n";
    std::cout << "Rank = " << result.rank << "\n";
    std::cout << "Singular values = " << formatVector(result.singularValues) << "\n";
    printMatrix(result.U, "U");
    printMatrix(result.Sigma, "Sigma");
    printMatrix(result.Vt, "V^T");
}

void handlePseudoInverse() {
    const Matrix A = readMatrix("A");
    printMatrix(pseudoInverse(A), "A^(+)");
}

void printMenu() {
    std::cout << "\n=== Linear Algebra Matrix Calculator ===\n";
    std::cout << " 1. Add two matrices\n";
    std::cout << " 2. Subtract two matrices\n";
    std::cout << " 3. Multiply a matrix by a scalar\n";
    std::cout << " 4. Multiply two matrices\n";
    std::cout << " 5. Transpose a matrix\n";
    std::cout << " 6. Dot product of two vectors\n";
    std::cout << " 7. Row echelon form (REF)\n";
    std::cout << " 8. Reduced row echelon form (RREF)\n";
    std::cout << " 9. Determinant\n";
    std::cout << "10. Inverse\n";
    std::cout << "11. Rank\n";
    std::cout << "12. Solve Ax = b\n";
    std::cout << "13. Bases for the row space, column space, and null space\n";
    std::cout << "14. PLU factorization\n";
    std::cout << "15. Gram-Schmidt / reduced QR factorization\n";
    std::cout << "16. Orthogonal projection onto Col(A)\n";
    std::cout << "17. Least-squares solution\n";
    std::cout << "18. Characteristic polynomial\n";
    std::cout << "19. Dominant eigenpair (power method)\n";
    std::cout << "20. Symmetric eigendecomposition\n";
    std::cout << "21. Compact singular value decomposition (SVD)\n";
    std::cout << "22. Pseudoinverse\n";
    std::cout << " 0. Exit\n";
}

}  // namespace

void runCalculator() {
    std::cout << "Matrix Calculator for Linear Algebra\n";
    std::cout << "This CLI covers the main computational topics from Lay's linear algebra text.\n";

    while (true) {
        printMenu();
        const int choice = readIntInRange("Choose an option: ", 0, 22);

        if (choice == 0) {
            std::cout << "Good luck with your linear algebra work.\n";
            break;
        }

        try {
            switch (choice) {
                case 1:
                    handleAddition();
                    break;
                case 2:
                    handleSubtraction();
                    break;
                case 3:
                    handleScalarMultiplication();
                    break;
                case 4:
                    handleMatrixMultiplication();
                    break;
                case 5:
                    handleTranspose();
                    break;
                case 6:
                    handleDotProduct();
                    break;
                case 7:
                    handleRef();
                    break;
                case 8:
                    handleRref();
                    break;
                case 9:
                    handleDeterminant();
                    break;
                case 10:
                    handleInverse();
                    break;
                case 11:
                    handleRank();
                    break;
                case 12:
                    handleSolveLinearSystem();
                    break;
                case 13:
                    handleSpaces();
                    break;
                case 14:
                    handlePlu();
                    break;
                case 15:
                    handleQr();
                    break;
                case 16:
                    handleProjection();
                    break;
                case 17:
                    handleLeastSquares();
                    break;
                case 18:
                    handleCharacteristicPolynomial();
                    break;
                case 19:
                    handlePowerMethod();
                    break;
                case 20:
                    handleSymmetricEigen();
                    break;
                case 21:
                    handleSvd();
                    break;
                case 22:
                    handlePseudoInverse();
                    break;
                default:
                    std::cout << "Unknown option.\n";
                    break;
            }
        } catch (const std::exception& error) {
            std::cout << "Error: " << error.what() << "\n";
        }
    }
}

}  // namespace matrix_solver
