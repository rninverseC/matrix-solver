#include "matrix_solver/operations.hpp"

#include "matrix_solver/io.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace matrix_solver {

namespace {

ld trace(const Matrix& matrix) {
    if (!matrix.isSquare()) {
        throw std::runtime_error("Trace is defined only for square matrices.");
    }
    ld total = 0.0L;
    for (int i = 0; i < matrix.rows; ++i) {
        total += matrix[i][i];
    }
    return cleanValue(total);
}

}  // namespace

std::vector<ld> getColumn(const Matrix& matrix, int column) {
    std::vector<ld> values(matrix.rows, 0.0L);
    for (int i = 0; i < matrix.rows; ++i) {
        values[i] = matrix[i][column];
    }
    return values;
}

void setColumn(Matrix& matrix, int column, const std::vector<ld>& values) {
    if (matrix.rows != static_cast<int>(values.size())) {
        throw std::runtime_error("Column size does not match matrix row count.");
    }
    for (int i = 0; i < matrix.rows; ++i) {
        matrix[i][column] = values[i];
    }
}

Matrix fromColumns(const std::vector<std::vector<ld>>& columns, int rowCount) {
    Matrix result(rowCount, static_cast<int>(columns.size()), 0.0L);
    for (int j = 0; j < result.cols; ++j) {
        if (static_cast<int>(columns[j].size()) != rowCount) {
            throw std::runtime_error("Column length mismatch while forming a matrix.");
        }
        setColumn(result, j, columns[j]);
    }
    return result;
}

ld dotProduct(const std::vector<ld>& a, const std::vector<ld>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same length for a dot product.");
    }
    ld sum = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return cleanValue(sum);
}

ld vectorNorm(const std::vector<ld>& values) {
    return std::sqrt(std::max<ld>(0.0L, dotProduct(values, values)));
}

std::vector<ld> scaleVector(const std::vector<ld>& values, ld scalar) {
    std::vector<ld> result(values.size(), 0.0L);
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = values[i] * scalar;
    }
    cleanVector(result);
    return result;
}

std::vector<ld> addVectors(const std::vector<ld>& a, const std::vector<ld>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same length.");
    }
    std::vector<ld> result(a.size(), 0.0L);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    cleanVector(result);
    return result;
}

std::vector<ld> subtractVectors(const std::vector<ld>& a, const std::vector<ld>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same length.");
    }
    std::vector<ld> result(a.size(), 0.0L);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    cleanVector(result);
    return result;
}

Matrix addMatrices(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::runtime_error("Matrix sizes must match for addition.");
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
        throw std::runtime_error("Matrix sizes must match for subtraction.");
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
        throw std::runtime_error("Inner dimensions must match for matrix multiplication.");
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

std::vector<ld> multiplyMatrixVector(const Matrix& matrix, const std::vector<ld>& vector) {
    if (matrix.cols != static_cast<int>(vector.size())) {
        throw std::runtime_error("Matrix and vector dimensions do not match.");
    }
    std::vector<ld> result(matrix.rows, 0.0L);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    cleanVector(result);
    return result;
}

Matrix augmentMatrices(const Matrix& left, const Matrix& right) {
    if (left.rows != right.rows) {
        throw std::runtime_error("Matrices must have the same number of rows to be augmented.");
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

EliminationResult toRef(Matrix matrix) {
    EliminationResult result;
    int pivotRow = 0;
    int swapCount = 0;

    for (int col = 0; col < matrix.cols && pivotRow < matrix.rows; ++col) {
        int bestRow = pivotRow;
        ld bestValue = 0.0L;
        for (int row = pivotRow; row < matrix.rows; ++row) {
            const ld candidate = std::fabsl(matrix[row][col]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }

        if (bestValue < EPS) {
            continue;
        }

        if (bestRow != pivotRow) {
            std::swap(matrix[bestRow], matrix[pivotRow]);
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
            const ld candidate = std::fabsl(matrix[row][col]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }

        if (bestValue < EPS) {
            continue;
        }

        if (bestRow != pivotRow) {
            std::swap(matrix[bestRow], matrix[pivotRow]);
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
        throw std::runtime_error("Determinants are defined only for square matrices.");
    }

    Matrix working = matrix;
    int sign = 1;

    for (int col = 0; col < working.cols; ++col) {
        int bestRow = col;
        ld bestValue = 0.0L;
        for (int row = col; row < working.rows; ++row) {
            const ld candidate = std::fabsl(working[row][col]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }

        if (bestValue < EPS) {
            return 0.0L;
        }

        if (bestRow != col) {
            std::swap(working[bestRow], working[col]);
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
        throw std::runtime_error("Only square matrices can be inverted.");
    }

    Matrix augmented = augmentMatrices(matrix, Matrix::identity(matrix.rows));
    const Matrix reduced = toRref(augmented).matrix;

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            const ld expected = (i == j) ? 1.0L : 0.0L;
            if (!nearlyEqual(reduced[i][j], expected)) {
                throw std::runtime_error("This matrix is singular, so it has no inverse.");
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

std::vector<std::vector<ld>> rowSpaceBasis(const Matrix& matrix) {
    const auto ref = toRef(matrix).matrix;
    std::vector<std::vector<ld>> basis;
    for (int i = 0; i < ref.rows; ++i) {
        if (!isZeroRow(ref[i])) {
            basis.push_back(ref[i]);
        }
    }
    return basis;
}

std::vector<std::vector<ld>> columnSpaceBasis(const Matrix& matrix) {
    const auto reduced = toRref(matrix);
    std::vector<std::vector<ld>> basis;
    for (int pivotColumn : reduced.pivotColumns) {
        basis.push_back(getColumn(matrix, pivotColumn));
    }
    return basis;
}

std::vector<std::vector<ld>> nullSpaceBasis(const Matrix& matrix) {
    const auto reduced = toRref(matrix);
    std::vector<bool> isPivotColumn(matrix.cols, false);
    for (int pivotColumn : reduced.pivotColumns) {
        if (pivotColumn < matrix.cols) {
            isPivotColumn[pivotColumn] = true;
        }
    }

    std::vector<std::vector<ld>> basis;
    for (int freeColumn = 0; freeColumn < matrix.cols; ++freeColumn) {
        if (isPivotColumn[freeColumn]) {
            continue;
        }

        std::vector<ld> vector(matrix.cols, 0.0L);
        vector[freeColumn] = 1.0L;
        for (int pivotRow = 0; pivotRow < reduced.rank; ++pivotRow) {
            const int pivotColumn = reduced.pivotColumns[pivotRow];
            if (pivotColumn < matrix.cols) {
                vector[pivotColumn] = -reduced.matrix[pivotRow][freeColumn];
            }
        }
        cleanVector(vector);
        basis.push_back(vector);
    }
    return basis;
}

LinearSystemResult solveLinearSystem(const Matrix& A, const std::vector<ld>& b) {
    if (A.rows != static_cast<int>(b.size())) {
        throw std::runtime_error("The right-hand side vector must have the same number of rows as A.");
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
        result.nullSpaceBasis = nullSpaceBasis(A);
    }
    return result;
}

LUResult pluDecomposition(const Matrix& matrix) {
    if (!matrix.isSquare()) {
        return {false, "PLU factorization requires a square matrix.", {}, {}, {}};
    }

    const int n = matrix.rows;
    Matrix P = Matrix::identity(n);
    Matrix L = Matrix::identity(n);
    Matrix U = matrix;

    for (int k = 0; k < n; ++k) {
        int pivotRow = k;
        ld bestValue = 0.0L;
        for (int row = k; row < n; ++row) {
            const ld candidate = std::fabsl(U[row][k]);
            if (candidate > bestValue) {
                bestValue = candidate;
                pivotRow = row;
            }
        }

        if (bestValue < EPS) {
            return {false, "The matrix is singular, so this PLU routine cannot continue with a nonzero pivot.", {}, {}, {}};
        }

        if (pivotRow != k) {
            std::swap(U[pivotRow], U[k]);
            std::swap(P[pivotRow], P[k]);
            for (int column = 0; column < k; ++column) {
                std::swap(L[pivotRow][column], L[k][column]);
            }
        }

        for (int row = k + 1; row < n; ++row) {
            const ld factor = U[row][k] / U[k][k];
            L[row][k] = factor;
            for (int column = k; column < n; ++column) {
                U[row][column] -= factor * U[k][column];
            }
            U[row][k] = 0.0L;
        }
    }

    cleanMatrix(L);
    cleanMatrix(U);
    return {true, "Computed P, L, and U such that P*A = L*U.", P, L, U};
}

QRResult reducedQR(const Matrix& matrix) {
    if (matrix.rows == 0 || matrix.cols == 0) {
        return {true, "Reduced QR of an empty matrix.", 0, Matrix(matrix.rows, 0), Matrix(0, matrix.cols)};
    }

    const int m = matrix.rows;
    const int n = matrix.cols;
    std::vector<std::vector<ld>> qColumns;
    Matrix fullR(n, n, 0.0L);

    int rank = 0;
    for (int j = 0; j < n; ++j) {
        auto working = getColumn(matrix, j);
        for (int i = 0; i < rank; ++i) {
            const ld coefficient = dotProduct(qColumns[i], working);
            fullR[i][j] = coefficient;
            working = subtractVectors(working, scaleVector(qColumns[i], coefficient));
        }

        const ld norm = vectorNorm(working);
        if (norm > EPS) {
            qColumns.push_back(scaleVector(working, 1.0L / norm));
            fullR[rank][j] = norm;
            ++rank;
        }
    }

    Matrix Q = fromColumns(qColumns, m);
    Matrix R(rank, n, 0.0L);
    for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < n; ++j) {
            R[i][j] = fullR[i][j];
        }
    }

    cleanMatrix(Q);
    cleanMatrix(R);
    return {true, "Computed a reduced QR factorization A = Q*R using modified Gram-Schmidt.", rank, Q, R};
}

std::vector<ld> projectOntoColumnSpace(const Matrix& A, const std::vector<ld>& b) {
    if (A.rows != static_cast<int>(b.size())) {
        throw std::runtime_error("The projection vector must have the same number of rows as A.");
    }

    const auto qr = reducedQR(A);
    std::vector<ld> projection(A.rows, 0.0L);
    for (int i = 0; i < qr.rank; ++i) {
        const auto q = getColumn(qr.Q, i);
        const ld coefficient = dotProduct(q, b);
        projection = addVectors(projection, scaleVector(q, coefficient));
    }
    cleanVector(projection);
    return projection;
}

bool isSymmetric(const Matrix& matrix, ld tolerance) {
    if (!matrix.isSquare()) {
        return false;
    }
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = i + 1; j < matrix.cols; ++j) {
            if (!nearlyEqual(matrix[i][j], matrix[j][i], tolerance)) {
                return false;
            }
        }
    }
    return true;
}

SymmetricEigenResult jacobiEigenDecomposition(const Matrix& matrix) {
    if (!matrix.isSquare()) {
        return {false, "Symmetric eigendecomposition requires a square matrix.", {}, {}};
    }
    if (!isSymmetric(matrix)) {
        return {false, "This Jacobi eigensolver is implemented for symmetric matrices only.", {}, {}};
    }

    const int n = matrix.rows;
    Matrix D = matrix;
    Matrix V = Matrix::identity(n);

    if (n == 1) {
        return {true, "Single-entry matrix: the entry itself is the eigenvalue.", {matrix[0][0]}, Matrix::identity(1)};
    }

    const int maxIterations = std::max(50, 200 * n * n);
    bool converged = false;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        int p = 0;
        int q = 1;
        ld largestOffDiagonal = 0.0L;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const ld candidate = std::fabsl(D[i][j]);
                if (candidate > largestOffDiagonal) {
                    largestOffDiagonal = candidate;
                    p = i;
                    q = j;
                }
            }
        }

        if (largestOffDiagonal < 1e-12L) {
            converged = true;
            break;
        }

        const ld theta = 0.5L * std::atan2(2.0L * D[p][q], D[q][q] - D[p][p]);
        const ld c = std::cos(theta);
        const ld s = std::sin(theta);

        const ld dpp = D[p][p];
        const ld dqq = D[q][q];
        const ld dpq = D[p][q];

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const ld dkp = D[k][p];
            const ld dkq = D[k][q];
            D[k][p] = D[p][k] = c * dkp - s * dkq;
            D[k][q] = D[q][k] = s * dkp + c * dkq;
        }

        D[p][p] = c * c * dpp - 2.0L * s * c * dpq + s * s * dqq;
        D[q][q] = s * s * dpp + 2.0L * s * c * dpq + c * c * dqq;
        D[p][q] = 0.0L;
        D[q][p] = 0.0L;

        for (int k = 0; k < n; ++k) {
            const ld vkp = V[k][p];
            const ld vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
        }
    }

    std::vector<ld> eigenvalues(n, 0.0L);
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = D[i][i];
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
        return eigenvalues[left] > eigenvalues[right];
    });

    std::vector<ld> sortedValues(n, 0.0L);
    Matrix sortedVectors(n, n, 0.0L);
    for (int newIndex = 0; newIndex < n; ++newIndex) {
        const int oldIndex = order[newIndex];
        sortedValues[newIndex] = cleanValue(eigenvalues[oldIndex]);
        setColumn(sortedVectors, newIndex, getColumn(V, oldIndex));
    }
    cleanMatrix(sortedVectors);

    return {
        true,
        converged ? "Jacobi iteration converged." : "Reached the Jacobi iteration limit; returning the latest approximation.",
        sortedValues,
        sortedVectors
    };
}

SVDResult compactSVD(const Matrix& matrix) {
    const Matrix AtA = multiplyMatrices(transpose(matrix), matrix);
    const auto eigen = jacobiEigenDecomposition(AtA);
    if (!eigen.success) {
        return {false, eigen.message, 0, {}, {}, {}, {}};
    }

    std::vector<ld> singularValues;
    std::vector<std::vector<ld>> uColumns;
    Matrix Vt(0, matrix.cols, 0.0L);

    for (int i = 0; i < static_cast<int>(eigen.eigenvalues.size()); ++i) {
        const ld lambda = std::max<ld>(0.0L, eigen.eigenvalues[i]);
        const ld sigma = std::sqrt(lambda);
        if (sigma < 1e-8L) {
            continue;
        }

        const auto v = getColumn(eigen.eigenvectors, i);
        auto u = scaleVector(multiplyMatrixVector(matrix, v), 1.0L / sigma);
        const ld norm = vectorNorm(u);
        if (norm < EPS) {
            continue;
        }
        u = scaleVector(u, 1.0L / norm);

        singularValues.push_back(cleanValue(sigma));
        uColumns.push_back(u);

        Matrix nextVt(static_cast<int>(singularValues.size()), matrix.cols, 0.0L);
        for (int row = 0; row < Vt.rows; ++row) {
            for (int col = 0; col < Vt.cols; ++col) {
                nextVt[row][col] = Vt[row][col];
            }
        }
        for (int col = 0; col < matrix.cols; ++col) {
            nextVt[nextVt.rows - 1][col] = v[col];
        }
        Vt = nextVt;
    }

    const int rank = static_cast<int>(singularValues.size());
    Matrix U = fromColumns(uColumns, matrix.rows);
    Matrix Sigma(rank, rank, 0.0L);
    for (int i = 0; i < rank; ++i) {
        Sigma[i][i] = singularValues[i];
    }

    cleanMatrix(U);
    cleanMatrix(Sigma);
    cleanMatrix(Vt);

    return {
        true,
        "Computed a compact SVD A = U*Sigma*V^T from the eigendecomposition of A^T*A.",
        rank,
        singularValues,
        U,
        Sigma,
        Vt
    };
}

Matrix pseudoInverse(const Matrix& matrix) {
    const auto svd = compactSVD(matrix);
    if (!svd.success) {
        throw std::runtime_error("Could not compute the pseudoinverse: " + svd.message);
    }

    Matrix pinv(matrix.cols, matrix.rows, 0.0L);
    for (int k = 0; k < svd.rank; ++k) {
        const ld sigma = svd.singularValues[k];
        const auto u = getColumn(svd.U, k);
        std::vector<ld> v(matrix.cols, 0.0L);
        for (int col = 0; col < matrix.cols; ++col) {
            v[col] = svd.Vt[k][col];
        }

        for (int i = 0; i < matrix.cols; ++i) {
            for (int j = 0; j < matrix.rows; ++j) {
                pinv[i][j] += (v[i] * u[j]) / sigma;
            }
        }
    }
    cleanMatrix(pinv);
    return pinv;
}

std::vector<ld> leastSquaresSolution(
    const Matrix& A,
    const std::vector<ld>& b,
    std::vector<ld>* projection,
    std::vector<ld>* residual) {
    if (A.rows != static_cast<int>(b.size())) {
        throw std::runtime_error("The right-hand side vector must have the same number of rows as A.");
    }

    const Matrix pinv = pseudoInverse(A);
    auto x = multiplyMatrixVector(pinv, b);
    if (projection != nullptr) {
        *projection = multiplyMatrixVector(A, x);
    }
    if (residual != nullptr) {
        std::vector<ld> localProjection;
        const std::vector<ld>& projectionVector =
            (projection != nullptr) ? *projection : (localProjection = multiplyMatrixVector(A, x));
        *residual = subtractVectors(b, projectionVector);
    }
    cleanVector(x);
    return x;
}

std::vector<ld> characteristicPolynomialCoefficients(const Matrix& matrix) {
    if (!matrix.isSquare()) {
        throw std::runtime_error("The characteristic polynomial is defined only for square matrices.");
    }

    const int n = matrix.rows;
    Matrix B = Matrix::identity(n);
    std::vector<ld> coefficients(n + 1, 0.0L);
    coefficients[0] = 1.0L;

    for (int k = 1; k <= n; ++k) {
        Matrix AB = multiplyMatrices(matrix, B);
        const ld coefficient = -trace(AB) / static_cast<ld>(k);
        coefficients[k] = cleanValue(coefficient);
        B = AB;
        for (int i = 0; i < n; ++i) {
            B[i][i] += coefficient;
        }
    }
    return coefficients;
}

std::string characteristicPolynomialToString(const std::vector<ld>& coefficients) {
    const int degree = static_cast<int>(coefficients.size()) - 1;
    std::ostringstream out;
    out << "lambda";
    if (degree > 1) {
        out << "^" << degree;
    }

    for (int i = 1; i <= degree; ++i) {
        const ld coefficient = coefficients[i];
        if (isZero(coefficient)) {
            continue;
        }

        out << (coefficient >= 0.0L ? " + " : " - ");
        const ld magnitude = std::fabsl(coefficient);
        const int power = degree - i;

        if (!(nearlyEqual(magnitude, 1.0L) && power > 0)) {
            out << formatNumber(magnitude);
            if (power > 0) {
                out << "*";
            }
        }

        if (power > 0) {
            out << "lambda";
            if (power > 1) {
                out << "^" << power;
            }
        }
    }

    out << " = 0";
    return out.str();
}

PowerMethodResult dominantEigenpair(const Matrix& matrix, int maxIterations, ld tolerance) {
    if (!matrix.isSquare()) {
        return {false, "The power method requires a square matrix.", 0, 0.0L, 0.0L, {}};
    }

    std::vector<ld> x(matrix.cols, 1.0L);
    x = scaleVector(x, 1.0L / vectorNorm(x));
    ld eigenvalue = 0.0L;

    for (int iteration = 1; iteration <= maxIterations; ++iteration) {
        const auto y = multiplyMatrixVector(matrix, x);
        const ld yNorm = vectorNorm(y);
        if (yNorm < EPS) {
            return {false, "The iterates collapsed to the zero vector, so the power method could not continue.", iteration, 0.0L, 0.0L, {}};
        }

        auto xNext = scaleVector(y, 1.0L / yNorm);
        const auto Ax = multiplyMatrixVector(matrix, xNext);
        const ld lambdaNext = dotProduct(xNext, Ax);
        const ld residualNorm = vectorNorm(subtractVectors(Ax, scaleVector(xNext, lambdaNext)));

        const ld delta = std::min(
            vectorNorm(subtractVectors(xNext, x)),
            vectorNorm(addVectors(xNext, x)));

        x = xNext;
        eigenvalue = lambdaNext;

        if (delta < tolerance && residualNorm < 1e-8L) {
            cleanVector(x);
            return {true, "Power method converged to the dominant eigenpair.", iteration, cleanValue(eigenvalue), cleanValue(residualNorm), x};
        }
    }

    cleanVector(x);
    const auto residual = vectorNorm(subtractVectors(
        multiplyMatrixVector(matrix, x),
        scaleVector(x, eigenvalue)));
    return {true, "Reached the iteration limit; returning the latest dominant-eigenpair approximation.", maxIterations, cleanValue(eigenvalue), cleanValue(residual), x};
}

}  // namespace matrix_solver
