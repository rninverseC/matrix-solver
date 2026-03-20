#include "matrix_solver/io.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace matrix_solver {

bool isZero(ld value, ld tolerance) {
    return std::fabsl(value) < tolerance;
}

ld cleanValue(ld value) {
    return isZero(value, DISPLAY_EPS) ? 0.0L : value;
}

bool nearlyEqual(ld a, ld b, ld tolerance) {
    return std::fabsl(a - b) < tolerance;
}

void cleanVector(std::vector<ld>& values) {
    for (auto& value : values) {
        value = cleanValue(value);
    }
}

void cleanMatrix(Matrix& matrix) {
    for (int i = 0; i < matrix.rows; ++i) {
        cleanVector(matrix[i]);
    }
}

std::string trimZeros(std::string value) {
    const auto decimal = value.find('.');
    if (decimal != std::string::npos) {
        while (!value.empty() && value.back() == '0') {
            value.pop_back();
        }
        if (!value.empty() && value.back() == '.') {
            value.pop_back();
        }
    }
    if (value == "-0") {
        value = "0";
    }
    return value;
}

std::string formatNumber(ld value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << cleanValue(value);
    return trimZeros(out.str());
}

bool isZeroRow(const std::vector<ld>& row) {
    for (ld value : row) {
        if (!isZero(value)) {
            return false;
        }
    }
    return true;
}

std::string formatVector(const std::vector<ld>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << formatNumber(values[i]);
    }
    out << "]";
    return out.str();
}

void printVector(const std::vector<ld>& values, const std::string& label) {
    std::cout << label << " = " << formatVector(values) << "\n";
}

void printMatrix(const Matrix& matrix, const std::string& label) {
    std::cout << label << " (" << matrix.rows << " x " << matrix.cols << "):\n";
    if (matrix.rows == 0 || matrix.cols == 0) {
        std::cout << "[empty matrix]\n";
        return;
    }

    std::vector<std::vector<std::string>> rendered(
        matrix.rows, std::vector<std::string>(matrix.cols));
    std::vector<std::size_t> widths(matrix.cols, 0);

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            rendered[i][j] = formatNumber(matrix[i][j]);
            widths[j] = std::max(widths[j], rendered[i][j].size());
        }
    }

    for (int i = 0; i < matrix.rows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < matrix.cols; ++j) {
            std::cout << std::setw(static_cast<int>(widths[j])) << rendered[i][j];
            if (j + 1 < matrix.cols) {
                std::cout << "  ";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "]\n";
    }
}

void printBasis(const std::vector<std::vector<ld>>& basis, const std::string& label) {
    std::cout << label << " basis:\n";
    if (basis.empty()) {
        std::cout << "  {0}\n";
        return;
    }
    for (std::size_t i = 0; i < basis.size(); ++i) {
        std::cout << "  v" << (i + 1) << " = " << formatVector(basis[i]) << "\n";
    }
}

int readPositiveInt(const std::string& prompt) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Input stream ended unexpectedly.");
        }

        std::istringstream iss(line);
        int value = 0;
        char extra = '\0';
        if (iss >> value && !(iss >> extra) && value > 0) {
            return value;
        }
        std::cout << "Please enter a positive integer.\n";
    }
}

int readIntInRange(const std::string& prompt, int minValue, int maxValue) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Input stream ended unexpectedly.");
        }

        std::istringstream iss(line);
        int value = 0;
        char extra = '\0';
        if (iss >> value && !(iss >> extra) && value >= minValue && value <= maxValue) {
            return value;
        }
        std::cout << "Please enter an integer from " << minValue << " to " << maxValue << ".\n";
    }
}

ld readLongDouble(const std::string& prompt) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Input stream ended unexpectedly.");
        }

        std::istringstream iss(line);
        ld value = 0.0L;
        char extra = '\0';
        if (iss >> value && !(iss >> extra)) {
            return value;
        }
        std::cout << "Please enter a valid number.\n";
    }
}

std::vector<ld> readEntries(const std::string& prompt, int count) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Input stream ended unexpectedly.");
        }

        std::istringstream iss(line);
        std::vector<ld> values;
        ld value = 0.0L;
        while (iss >> value) {
            values.push_back(value);
        }

        iss.clear();
        iss >> std::ws;
        if (static_cast<int>(values.size()) == count && iss.eof()) {
            return values;
        }
        std::cout << "Please enter exactly " << count << " numbers separated by spaces.\n";
    }
}

Matrix readMatrix(const std::string& name) {
    const int rows = readPositiveInt("Rows of " + name + ": ");
    const int cols = readPositiveInt("Columns of " + name + ": ");
    Matrix matrix(rows, cols, 0.0L);

    std::cout << "Enter " << name << " row by row.\n";
    for (int i = 0; i < rows; ++i) {
        auto row = readEntries("  row " + std::to_string(i + 1) + ": ", cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = row[j];
        }
    }
    return matrix;
}

std::vector<ld> readVector(const std::string& name, int length) {
    std::cout << "Enter vector " << name << " with " << length << " entries.\n";
    return readEntries("  values: ", length);
}

}  // namespace matrix_solver
