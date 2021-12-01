//
// Created by kabanpunk on 22.11.2021.
//

#ifndef LAB7_MATRIX_H
#define LAB7_MATRIX_H

#include <iostream>

class Matrix {
public:
    explicit Matrix(int);
    Matrix(int, int, int);
    Matrix(float*, int);
    Matrix();
    ~Matrix();
    Matrix(const Matrix&);
    Matrix& operator=(const Matrix&);

    inline float& operator()(int x, int y) { return p[x * N_ + y]; }

    Matrix& operator+=(const Matrix&);
    Matrix& operator-=(const Matrix&);
    Matrix& operator*=(const Matrix&);
    Matrix& operator*=(float);
    Matrix& operator/=(float);
    Matrix  operator^(int);

    friend std::ostream& operator<<(std::ostream&, const Matrix&);
    friend std::istream& operator>>(std::istream&, Matrix&);

    Matrix transpose();
    float getMaxRowsSum();
    float getMaxCollumnsSum();

    static Matrix createIdentity(int);

private:
    int N_;
    float *p;

    static float round_(float, int);
    void allocSpace();
};

Matrix operator+(const Matrix&, const Matrix&);
Matrix operator-(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, float);
Matrix operator*(float, const Matrix&);
Matrix operator/(const Matrix&, float);

#endif //LAB7_MATRIX_H
