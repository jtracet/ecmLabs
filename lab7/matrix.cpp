//
// Created by kabanpunk on 22.11.2021.
//

#include <stdexcept>
#include <bits/stdc++.h>
#include "math.h"
#include "matrix.h"


Matrix::Matrix(int N) : N_(N)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = 0;
        }
    }
}

Matrix::Matrix(int N, int a, int b) : N_(N)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = rand() % (b - a) + a;
        }
    }
}

Matrix::Matrix(float* a, int N) : N_(N)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = a[i * N_ + j];
        }
    }
}

Matrix::Matrix() : N_(1)
{
    allocSpace();
    p[0] = 0;
}

Matrix::~Matrix()
{
    delete[] p;
}

Matrix::Matrix(const Matrix& m) : N_(m.N_)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = m.p[i * N_ + j];
        }
    }
}

Matrix& Matrix::operator=(const Matrix& m)
{
    if (this == &m) {
        return *this;
    }

    if (N_ != m.N_) {
        free(p);

        N_ = m.N_;
        allocSpace();
    }

    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = m.p[i * N_ + j];
        }
    }
    return *this;
}

Matrix& Matrix::operator+=(const Matrix& m)
{
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] += m.p[i * N_ + j];
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& m)
{
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] -= m.p[i * N_ + j];
        }
    }
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& m)
{
    Matrix temp(N_);
    for (int i = 0; i < temp.N_; ++i) {
        for (int j = 0; j < temp.N_; ++j) {
            for (int k = 0; k < N_; ++k) {
                temp.p[i * temp.N_ + j] += (p[i * N_ + k] * m.p[k * N_ + j]);
            }
        }
    }
    return (*this = temp);
}

Matrix& Matrix::operator*=(float num)
{
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] *= num;
        }
    }
    return *this;
}

Matrix& Matrix::operator/=(float num)
{
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] /= num;
        }
    }
    return *this;
}

Matrix Matrix::transpose()
{
    Matrix ret(N_);
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            ret.p[j * N_ + i] = p[i * N_ + j];
        }
    }
    return ret;
}

float Matrix::getMaxRowsSum()
{
    float maxSum = INT_MIN;
    for(int i = 0; i < N_; ++i)
    {
        float s = 0;
        for(int j = 0; j < N_; ++j)
        {
            s += fabs(p[i * N_ + j]);
        }
        maxSum = std::max(maxSum, s);
    }
    return maxSum;
}

float Matrix::getMaxCollumnsSum()
{
    float maxSum = INT_MIN;
    for (int i = 0; i < N_; ++i)
    {
        float s = 0;
        for (int j = 0; j < N_; ++j)
        {
            s += fabs(p[j * N_ + i]);
        }
        maxSum = std::max(maxSum, s);
    }
    return maxSum;
}

Matrix Matrix::createIdentity(int N)
{
    Matrix temp(N);
    for (int i = 0; i < temp.N_; ++i) {
        for (int j = 0; j < temp.N_; ++j) {
            if (i == j) {
                temp.p[i * temp.N_ + j] = 1;
            } else {
                temp.p[i * temp.N_ + j] = 0;
            }
        }
    }
    return temp;
}

void Matrix::allocSpace()
{
    p = new float[N_ * N_];
}

Matrix operator+(const Matrix& m1, const Matrix& m2)
{
    Matrix temp(m1);
    return (temp += m2);
}

Matrix operator-(const Matrix& m1, const Matrix& m2)
{
    Matrix temp(m1);
    return (temp -= m2);
}

Matrix operator*(const Matrix& m1, const Matrix& m2)
{
    Matrix temp(m1);
    return (temp *= m2);
}

Matrix operator*(const Matrix& m, float num)
{
    Matrix temp(m);
    return (temp *= num);
}

Matrix operator*(float num, const Matrix& m)
{
    return (m * num);
}

Matrix operator/(const Matrix& m, float num)
{
    Matrix temp(m);
    return (temp /= num);
}

std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    for (int i = 0; i < m.N_; ++i) {
        os << Matrix::round_( m.p[i * m.N_], 3);
        for (int j = 1; j < m.N_; ++j) {
            os << " " << Matrix::round_( m.p[i * m.N_ + j], 3);
        }
        os << std::endl;
    }
    return os;
}

std::istream& operator>>(std::istream& is, Matrix& m)
{
    for (int i = 0; i < m.N_; ++i) {
        for (int j = 0; j < m.N_; ++j) {
            is >> m.p[i * m.N_ + j];
        }
    }
    return is;
}

float Matrix::round_(float x, int pr) {
    float d = pow(10, pr);
    return round(x * d ) / d + 0.0;
}
