#pragma once

#include "matrix_solver/types.hpp"

#include <string>
#include <vector>

using namespace std;

namespace matrix_solver {

bool isZero(ld value, ld tolerance = EPS);
ld cleanValue(ld value);
bool nearlyEqual(ld a, ld b, ld tolerance = 1e-8L);

void cleanVector(vector<ld>& values);
void cleanMatrix(Matrix& matrix);

string trimZeros(string value);
string formatNumber(ld value);
bool isZeroRow(const vector<ld>& row);
string formatVector(const vector<ld>& values);

void printVector(const vector<ld>& values, const string& label);
void printMatrix(const Matrix& matrix, const string& label);
void printBasis(const vector<vector<ld>>& basis, const string& label);

int readPositiveInt(const string& prompt);
int readIntInRange(const string& prompt, int minValue, int maxValue);
ld readLongDouble(const string& prompt);
vector<ld> readEntries(const string& prompt, int count);
Matrix readMatrix(const string& name);
vector<ld> readVector(const string& name, int length);

}  // namespace matrix_solver
