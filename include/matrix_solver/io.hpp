#pragma once

#include "matrix_solver/types.hpp"

#include <string>
#include <vector>

namespace matrix_solver {

bool isZero(ld value, ld tolerance = EPS);
ld cleanValue(ld value);
bool nearlyEqual(ld a, ld b, ld tolerance = 1e-8L);

void cleanVector(std::vector<ld>& values);
void cleanMatrix(Matrix& matrix);

std::string trimZeros(std::string value);
std::string formatNumber(ld value);
bool isZeroRow(const std::vector<ld>& row);
std::string formatVector(const std::vector<ld>& values);

void printVector(const std::vector<ld>& values, const std::string& label);
void printMatrix(const Matrix& matrix, const std::string& label);
void printBasis(const std::vector<std::vector<ld>>& basis, const std::string& label);

int readPositiveInt(const std::string& prompt);
int readIntInRange(const std::string& prompt, int minValue, int maxValue);
ld readLongDouble(const std::string& prompt);
std::vector<ld> readEntries(const std::string& prompt, int count);
Matrix readMatrix(const std::string& name);
std::vector<ld> readVector(const std::string& name, int length);

}  // namespace matrix_solver
