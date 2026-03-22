#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

namespace matrix_solver {

using ld = long double;

constexpr ld EPS = 1e-10L;
constexpr ld DISPLAY_EPS = 1e-9L;

struct Matrix {
    int rows = 0;
    int cols = 0;
    vector<vector<ld>> data;

    Matrix() = default;

    Matrix(int r, int c, ld value = 0.0L) : rows(r), cols(c), data(r, vector<ld>(c, value)) {
        if (r < 0 || c < 0) {
            throw runtime_error("Matrix dimensions must be non-negative.");
        }
    }

    explicit Matrix(vector<vector<ld>> values): 
          rows(static_cast<int>(values.size())),
          cols(values.empty() ? 0 : static_cast<int>(values.front().size())),
          data(std::move(values)) {
        for (const auto& row : data) {
            if (static_cast<int>(row.size()) != cols) {
                throw runtime_error("All rows in a matrix must have the same length.");
            }
        }
    }

    vector<ld>& operator[](int row) { return data[row]; }
    const vector<ld>& operator[](int row) const { return data[row]; }

    bool isSquare() const { return rows == cols; }

    static Matrix identity(int n) {
        if (n < 0) {
            throw runtime_error("Identity matrix size must be non-negative.");
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
    vector<int> pivotColumns;
    int rank = 0;
    int swapCount = 0;
};

struct LinearSystemResult {
    bool consistent = true;
    bool unique = false;
    int rankA = 0;
    vector<ld> particularSolution;
    vector<vector<ld>> nullSpaceBasis;
    Matrix augmentedRref;
};

}  // namespace matrix_solver
