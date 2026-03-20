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
    return fabs(value) < tolerance;
}

ld cleanValue(ld value) {
    return isZero(value, DISPLAY_EPS) ? 0.0L : value;
}

bool nearlyEqual(ld a, ld b, ld tolerance) {
    return fabs(a - b) < tolerance;
}

void cleanVector(vector<ld>& values) {
    for (auto& value : values) {
        value = cleanValue(value);
    }
}

void cleanMatrix(Matrix& matrix) {
    for (int i = 0; i < matrix.rows; ++i) {
        cleanVector(matrix[i]);
    }
}

string trimZeros(string value) {
    const auto decimal = value.find('.');
    if (decimal != string::npos) {
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

string formatNumber(ld value) {
    ostringstream out;
    out << fixed << setprecision(6) << cleanValue(value);
    return trimZeros(out.str());
}

bool isZeroRow(const vector<ld>& row) {
    for (ld value : row) {
        if (!isZero(value)) {
            return false;
        }
    }
    return true;
}

string formatVector(const vector<ld>& values) {
    ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << formatNumber(values[i]);
    }
    out << "]";
    return out.str();
}

void printVector(const vector<ld>& values, const string& label) {
    cout << label << " = " << formatVector(values) << "\n";
}

void printMatrix(const Matrix& matrix, const string& label) {
    cout << label << " (" << matrix.rows << " x " << matrix.cols << "):\n";
    if (matrix.rows == 0 || matrix.cols == 0) {
        cout << "[empty matrix]\n";
        return;
    }

    vector<vector<string>> rendered(matrix.rows, vector<string>(matrix.cols));
    vector<size_t> widths(matrix.cols, 0);

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            rendered[i][j] = formatNumber(matrix[i][j]);
            widths[j] = max(widths[j], rendered[i][j].size());
        }
    }

    for (int i = 0; i < matrix.rows; ++i) {
        cout << "[ ";
        for (int j = 0; j < matrix.cols; ++j) {
            cout << setw(static_cast<int>(widths[j])) << rendered[i][j];
            if (j + 1 < matrix.cols) {
                cout << "  ";
            } else {
                cout << " ";
            }
        }
        cout << "]\n";
    }
}

void printBasis(const vector<vector<ld>>& basis, const string& label) {
    cout << label << " basis:\n";
    if (basis.empty()) {
        cout << "  {0}\n";
        return;
    }
    for (size_t i = 0; i < basis.size(); ++i) {
        cout << "  v" << (i + 1) << " = " << formatVector(basis[i]) << "\n";
    }
}

int readPositiveInt(const string& prompt) {
    while (true) {
        cout << prompt;
        string line;
        if (!getline(cin, line)) {
            throw runtime_error("Input stream ended unexpectedly.");
        }

        istringstream iss(line);
        int value = 0;
        char extra = '\0';
        if (iss >> value && !(iss >> extra) && value > 0) {
            return value;
        }
        cout << "Please enter a positive integer.\n";
    }
}

int readIntInRange(const string& prompt, int minValue, int maxValue) {
    while (true) {
        cout << prompt;
        string line;
        if (!getline(cin, line)) {
            throw runtime_error("Input stream ended unexpectedly.");
        }

        istringstream iss(line);
        int value = 0;
        char extra = '\0';
        if (iss >> value && !(iss >> extra) && value >= minValue && value <= maxValue) {
            return value;
        }
        cout << "Please enter an integer from " << minValue << " to " << maxValue << ".\n";
    }
}

ld readLongDouble(const string& prompt) {
    while (true) {
        cout << prompt;
        string line;
        if (!getline(cin, line)) {
            throw runtime_error("Input stream ended unexpectedly.");
        }

        istringstream iss(line);
        ld value = 0.0L;
        char extra = '\0';
        if (iss >> value && !(iss >> extra)) {
            return value;
        }
        cout << "Please enter a valid number.\n";
    }
}

vector<ld> readEntries(const string& prompt, int count) {
    while (true) {
        cout << prompt;
        string line;
        if (!getline(cin, line)) {
            throw runtime_error("Input stream ended unexpectedly.");
        }

        istringstream iss(line);
        vector<ld> values;
        ld value = 0.0L;
        while (iss >> value) {
            values.push_back(value);
        }

        iss.clear();
        iss >> ws;
        if (static_cast<int>(values.size()) == count && iss.eof()) {
            return values;
        }
        cout << "Please enter exactly " << count << " numbers separated by spaces.\n";
    }
}

Matrix readMatrix(const string& name) {
    const int rows = readPositiveInt("Rows of " + name + ": ");
    const int cols = readPositiveInt("Columns of " + name + ": ");
    Matrix matrix(rows, cols, 0.0L);

    cout << "Enter " << name << " row by row.\n";
    for (int i = 0; i < rows; ++i) {
        auto row = readEntries("  row " + to_string(i + 1) + ": ", cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = row[j];
        }
    }
    return matrix;
}

vector<ld> readVector(const string& name, int length) {
    cout << "Enter vector " << name << " with " << length << " entries.\n";
    return readEntries("  values: ", length);
}

}  // namespace matrix_solver
