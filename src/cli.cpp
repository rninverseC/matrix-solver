#include "matrix_solver/cli.hpp"

#include "matrix_solver/io.hpp"
#include "matrix_solver/operations.hpp"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

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

void printMenu() {
    std::cout << "\n=== Ultimate Calculator ===\n";
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
    std::cout << " 0. Exit\n";
}

}  // namespace

void runCalculator() {
    std::cout << "Matrix Calculator for Linear Algebra\n";
    std::cout << "This CLI now includes only the 12 core matrix operations you asked for.\n";

    while (true) {
        printMenu();
        const int choice = readIntInRange("Choose an option: ", 0, 12);

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
