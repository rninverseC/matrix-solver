#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace matrix_solver {

using ld = long double;

constexpr ld EPS = 1e-10L;
constexpr ld DISPLAY_EPS = 1e-9L;

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<std::vector<ld>> data;

    Matrix() = default;

    Matrix(int r, int c, ld value = 0.0L) : rows(r), cols(c), data(r, std::vector<ld>(c, value)) {
        if (r < 0 || c < 0) {
            throw std::runtime_error("Matrix dimensions must be non-negative.");
        }
    }

    explicit Matrix(std::vector<std::vector<ld>> values)
        : rows(static_cast<int>(values.size())),
          cols(values.empty() ? 0 : static_cast<int>(values.front().size())),
          data(std::move(values)) {
        for (const auto& row : data) {
            if (static_cast<int>(row.size()) != cols) {
                throw std::runtime_error("All rows in a matrix must have the same length.");
            }
        }
    }

    std::vector<ld>& operator[](int row) { return data[row]; }
    const std::vector<ld>& operator[](int row) const { return data[row]; }

    bool isSquare() const { return rows == cols; }

    static Matrix identity(int n) {
        if (n < 0) {
            throw std::runtime_error("Identity matrix size must be non-negative.");
        }
        Matrix result(n, n, 0.0L);
        for (int i = 0; i < n; ++i) {
            result[i][i] = 1.0L;
        }
        return result;
    }
};

struct EliminationResult {
    Matrix matrix;
    std::vector<int> pivotColumns;
    int rank = 0;
    int swapCount = 0;
};

struct LinearSystemResult {
    bool consistent = true;
    bool unique = false;
    int rankA = 0;
    std::vector<ld> particularSolution;
    std::vector<std::vector<ld>> nullSpaceBasis;
    Matrix augmentedRref;
};

struct LUResult {
    bool success = false;
    std::string message;
    Matrix P;
    Matrix L;
    Matrix U;
};

struct QRResult {
    bool success = false;
    std::string message;
    int rank = 0;
    Matrix Q;
    Matrix R;
};

struct PowerMethodResult {
    bool success = false;
    std::string message;
    int iterations = 0;
    ld eigenvalue = 0.0L;
    ld residualNorm = 0.0L;
    std::vector<ld> eigenvector;
};

struct SymmetricEigenResult {
    bool success = false;
    std::string message;
    std::vector<ld> eigenvalues;
    Matrix eigenvectors;
};

struct SVDResult {
    bool success = false;
    std::string message;
    int rank = 0;
    std::vector<ld> singularValues;
    Matrix U;
    Matrix Sigma;
    Matrix Vt;
};

}  // namespace matrix_solver
