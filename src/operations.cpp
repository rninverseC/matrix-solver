#include "matrix_solver/operations.hpp"

#include "matrix_solver/io.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace std;

namespace matrix_solver {

namespace {

Matrix augmentMatrices(const Matrix& left, const Matrix& right) {
    if (left.rows != right.rows) {
        throw runtime_error("Matrices must have the same number of rows to be augmented.");
    }

    Matrix result(left.rows, left.cols + right.cols, 0.0L);
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < left.cols; ++j) {
            result[i][j] = left[i][j];
        }
        for (int j = 0; j < right.cols; ++j) {
            result[i][left.cols + j] = right[i][j];
        }
    }
    return result;
}

vector<vector<ld>> buildNullSpaceBasis(const Matrix& matrix) {
    const auto reduced = toRref(matrix);
    vector<bool> isPivotColumn(matrix.cols, false);

    for (int pivotColumn : reduced.pivotColumns) {
        if (pivotColumn < matrix.cols) {
            isPivotColumn[pivotColumn] = true;
        }
    }

    vector<vector<ld>> basis;
    for (int freeColumn = 0; freeColumn < matrix.cols; ++freeColumn) {
        if (isPivotColumn[freeColumn]) {
            continue;
        }

        vector<ld> basisVector(matrix.cols, 0.0L);
        basisVector[freeColumn] = 1.0L;

        for (int pivotRow = 0; pivotRow < reduced.rank; ++pivotRow) {
            const int pivotColumn = reduced.pivotColumns[pivotRow];
            if (pivotColumn < matrix.cols) {
                basisVector[pivotColumn] = -reduced.matrix[pivotRow][freeColumn];
            }
        }

        cleanVector(basisVector);
        basis.push_back(basisVector);
    }

    return basis;
}

}  // namespace

ld dotProduct(const vector<ld>& a, const vector<ld>& b) {
    if (a.size() != b.size()) {
        throw runtime_error("Vectors must have the same length for a dot product.");
    }

    ld sum = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return cleanValue(sum);
}

Matrix addMatrices(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw runtime_error("Matrix sizes must match for addition.");
    }

    Matrix result(a.rows, a.cols, 0.0L);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    cleanMatrix(result);
    return result;
}

Matrix subtractMatrices(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw runtime_error("Matrix sizes must match for subtraction.");
    }

    Matrix result(a.rows, a.cols, 0.0L);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }

    cleanMatrix(result);
    return result;
}

Matrix scalarMultiply(const Matrix& matrix, ld scalar) {
    Matrix result(matrix.rows, matrix.cols, 0.0L);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }

    cleanMatrix(result);
    return result;
}

Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix.cols, matrix.rows, 0.0L);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    cleanMatrix(result);
    return result;
}

Matrix multiplyMatrices(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw runtime_error("Inner dimensions must match for matrix multiplication.");
    }

    Matrix result(a.rows, b.cols, 0.0L);
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < a.cols; ++k) {
            if (isZero(a[i][k])) {
                continue;
            }
            for (int j = 0; j < b.cols; ++j) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    cleanMatrix(result);
    return result;
}

EliminationResult toRef(Matrix matrix) {
    EliminationResult result;
    int pivotRow = 0;
    int swapCount = 0;

    for (int col = 0; col < matrix.cols && pivotRow < matrix.rows; ++col) {
        int bestRow = pivotRow;
        ld bestValue = 0.0L;

        for (int row = pivotRow; row < matrix.rows; ++row) {
            const ld candidate = fabs(matrix[row][col]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }

        if (bestValue < EPS) {
            continue;
        }

        if (bestRow != pivotRow) {
            swap(matrix[bestRow], matrix[pivotRow]);
            ++swapCount;
        }

        const ld pivotValue = matrix[pivotRow][col];
        for (int row = pivotRow + 1; row < matrix.rows; ++row) {
            if (isZero(matrix[row][col])) {
                continue;
            }

            const ld factor = matrix[row][col] / pivotValue;
            for (int current = col; current < matrix.cols; ++current) {
                matrix[row][current] -= factor * matrix[pivotRow][current];
            }
            matrix[row][col] = 0.0L;
        }

        result.pivotColumns.push_back(col);
        ++pivotRow;
    }

    cleanMatrix(matrix);
    result.matrix = matrix;
    result.rank = static_cast<int>(result.pivotColumns.size());
    result.swapCount = swapCount;
    return result;
}

EliminationResult toRref(Matrix matrix) {
    EliminationResult result;
    int pivotRow = 0;
    int swapCount = 0;

    for (int col = 0; col < matrix.cols && pivotRow < matrix.rows; ++col) {
        int bestRow = pivotRow;
        ld bestValue = 0.0L;

        for (int row = pivotRow; row < matrix.rows; ++row) {
            const ld candidate = fabs(matrix[row][col]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }

        if (bestValue < EPS) {
            continue;
        }

        if (bestRow != pivotRow) {
            swap(matrix[bestRow], matrix[pivotRow]);
            ++swapCount;
        }

        const ld pivotValue = matrix[pivotRow][col];
        for (int current = 0; current < matrix.cols; ++current) {
            matrix[pivotRow][current] /= pivotValue;
        }

        for (int row = 0; row < matrix.rows; ++row) {
            if (row == pivotRow || isZero(matrix[row][col])) {
                continue;
            }

            const ld factor = matrix[row][col];
            for (int current = 0; current < matrix.cols; ++current) {
                matrix[row][current] -= factor * matrix[pivotRow][current];
            }
            matrix[row][col] = 0.0L;
        }

        result.pivotColumns.push_back(col);
        ++pivotRow;
    }

    cleanMatrix(matrix);
    result.matrix = matrix;
    result.rank = static_cast<int>(result.pivotColumns.size());
    result.swapCount = swapCount;
    return result;
}

int rankOfMatrix(const Matrix& matrix) {
    return toRref(matrix).rank;
}

ld determinant(const Matrix& matrix) {
    if (!matrix.isSquare()) {
        throw runtime_error("Determinants are defined only for square matrices.");
    }

    Matrix working = matrix;
    int sign = 1;

    for (int col = 0; col < working.cols; ++col) {
        int bestRow = col;
        ld bestValue = 0.0L;

        for (int row = col; row < working.rows; ++row) {
            const ld candidate = fabs(working[row][col]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }

        if (bestValue < EPS) {
            return 0.0L;
        }

        if (bestRow != col) {
            swap(working[bestRow], working[col]);
            sign *= -1;
        }

        const ld pivotValue = working[col][col];
        for (int row = col + 1; row < working.rows; ++row) {
            const ld factor = working[row][col] / pivotValue;
            for (int current = col; current < working.cols; ++current) {
                working[row][current] -= factor * working[col][current];
            }
            working[row][col] = 0.0L;
        }
    }

    ld det = static_cast<ld>(sign);
    for (int i = 0; i < working.rows; ++i) {
        det *= working[i][i];
    }
    return cleanValue(det);
}

Matrix inverseMatrix(const Matrix& matrix) {
    if (!matrix.isSquare()) {
        throw runtime_error("Only square matrices can be inverted.");
    }

    const Matrix augmented = augmentMatrices(matrix, Matrix::identity(matrix.rows));
    const Matrix reduced = toRref(augmented).matrix;

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            const ld expected = (i == j) ? 1.0L : 0.0L;
            if (!nearlyEqual(reduced[i][j], expected)) {
                throw runtime_error("This matrix is singular, so it has no inverse.");
            }
        }
    }

    Matrix inverse(matrix.rows, matrix.cols, 0.0L);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            inverse[i][j] = reduced[i][matrix.cols + j];
        }
    }

    cleanMatrix(inverse);
    return inverse;
}

LinearSystemResult solveLinearSystem(const Matrix& A, const vector<ld>& b) {
    if (A.rows != static_cast<int>(b.size())) {
        throw runtime_error("The right-hand side vector must have the same number of rows as A.");
    }

    Matrix right(A.rows, 1, 0.0L);
    for (int i = 0; i < A.rows; ++i) {
        right[i][0] = b[i];
    }

    LinearSystemResult result;
    result.augmentedRref = toRref(augmentMatrices(A, right)).matrix;

    for (int row = 0; row < result.augmentedRref.rows; ++row) {
        bool allZero = true;
        for (int col = 0; col < A.cols; ++col) {
            if (!isZero(result.augmentedRref[row][col])) {
                allZero = false;
                break;
            }
        }

        if (allZero && !isZero(result.augmentedRref[row][A.cols])) {
            result.consistent = false;
            result.unique = false;
            return result;
        }
    }

    result.rankA = rankOfMatrix(A);
    result.unique = (result.rankA == A.cols);
    result.particularSolution.assign(A.cols, 0.0L);

    for (int row = 0; row < result.augmentedRref.rows; ++row) {
        int pivotColumn = -1;
        for (int col = 0; col < A.cols; ++col) {
            if (!isZero(result.augmentedRref[row][col])) {
                pivotColumn = col;
                break;
            }
        }

        if (pivotColumn != -1) {
            result.particularSolution[pivotColumn] = result.augmentedRref[row][A.cols];
        }
    }

    cleanVector(result.particularSolution);
    if (!result.unique) {
        result.nullSpaceBasis = buildNullSpaceBasis(A);
    }

    return result;
}

} 
