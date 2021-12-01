//
// Created by kabanpunk on 25.11.2021.
//

#include <immintrin.h>
#include <stdexcept>
#include <bits/stdc++.h>
#include "math.h"
#include "matrixSSE.h"

MatrixSSE::MatrixSSE(int N) : N_(N), NS(N * N)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = 0;
        }
    }
}

MatrixSSE::MatrixSSE(int N, int a, int b) : N_(N), NS(N * N)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = rand() % (b - a) + a;
        }
    }
}

MatrixSSE::MatrixSSE(float* a, int N) : N_(N), NS(N * N)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = a[i * N_ + j];
        }
    }
}

MatrixSSE::MatrixSSE() : N_(1), NS(1)
{
    allocSpace();
    p[0] = 0;
}

MatrixSSE::~MatrixSSE()
{
    delete[] p;
}

MatrixSSE::MatrixSSE(const MatrixSSE& m) : N_(m.N_), NS(m.N_ * m.N_)
{
    allocSpace();
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] = m.p[i * N_ + j];
        }
    }
}

MatrixSSE& MatrixSSE::operator=(const MatrixSSE& m)
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

MatrixSSE::Vector4 MatrixSSE::setVector(const MatrixSSE& m, int k)
{
    Vector4 v =
            {
                    ((k + 0) < m.NS) ? m.p[k + 0] : 0,
                    ((k + 1) < m.NS) ? m.p[k + 1] : 0,
                    ((k + 2) < m.NS) ? m.p[k + 2] : 0,
                    ((k + 3) < m.NS) ? m.p[k + 3] : 0
            };

    return v;
}

MatrixSSE& MatrixSSE::operator+=(const MatrixSSE& m)
{
    MatrixSSE temp(N_);
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; j += 4) {
            int l = i * N_ + j;
            Vector4 a = setVector(*this, l);
            Vector4 b = setVector(m, l);
            Vector4 res;
            SSE_Add(&res, &a, &b);

            temp.p[l + 0] = res.x;
            temp.p[l + 1] = res.y;
            temp.p[l + 2] = res.z;
            temp.p[l + 3] = res.w;
        }
    }
    return (*this = temp);
}

MatrixSSE& MatrixSSE::operator-=(const MatrixSSE& m)
{
    MatrixSSE temp(N_);
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; j += 4) {
            int l = i * N_ + j;
            Vector4 a = setVector(*this, l);
            Vector4 b = setVector(m, l);
            Vector4 res;
            SSE_Sub(&res, &a, &b);

            temp.p[l + 0] = res.x;
            temp.p[l + 1] = res.y;
            temp.p[l + 2] = res.z;
            temp.p[l + 3] = res.w;
        }
    }
    return (*this = temp);
}

MatrixSSE& MatrixSSE::operator*=(const MatrixSSE& m)
{
    MatrixSSE temp(N_);

    for (int i = 0; i < N_; ++i)
    {
        for (int j = 0; j < N_; ++j)
        {
            temp.p[i * N_ + j] = 0;
            for (int k = 0; k < N_ ; k += 4)
            {
                int l = i * N_ + k;
                Vector4 a =
                        {
                                ((l + 0) < NS) ? p[l + 0] : 0,
                                ((l + 1) < NS) ? p[l + 1] : 0,
                                ((l + 2) < NS) ? p[l + 2] : 0,
                                ((l + 3) < NS) ? p[l + 3] : 0,

                        };
                Vector4 b =
                        {
                                ((k + 0) * m.N_ + j < m.NS) ? m.p[(k + 0) * m.N_ + j] : 0,
                                ((k + 1) * m.N_ + j < m.NS) ? m.p[(k + 1) * m.N_ + j] : 0,
                                ((k + 2) * m.N_ + j < m.NS) ? m.p[(k + 2) * m.N_ + j] : 0,
                                ((k + 3) * m.N_ + j < m.NS) ? m.p[(k + 3) * m.N_ + j] : 0,
                        };
                Vector4 res;
                SSE_Mul(&res, &a, &b);
                temp.p[i * N_ + j] += res.x + res.y + res.z + res.w;
            }
        }
    }

    return (*this = temp);
}

MatrixSSE& MatrixSSE::operator*=(float num)
{
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] *= num;
        }
    }
    return *this;
}

MatrixSSE& MatrixSSE::operator/=(float num)
{
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            p[i * N_ + j] /= num;
        }
    }
    return *this;
}

MatrixSSE MatrixSSE::transpose()
{
    MatrixSSE ret(N_);
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            ret.p[j * N_ + i] = p[i * N_ + j];
        }
    }
    return ret;
}

float MatrixSSE::getMaxRowsSum()
{
    float maxSum = INT_MIN;
    for(int i = 0; i < N_; ++i)
    {
        float s = 0;
        for(int j = 0; j < N_; ++j)
        {
            s += fabs(p[j * N_ + i]);
        }
        maxSum = std::max(maxSum, s);
    }
    return maxSum;
}

float MatrixSSE::getMaxCollumnsSum()
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

MatrixSSE MatrixSSE::createIdentity(int N)
{
    MatrixSSE temp(N);
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

void MatrixSSE::SSE_Add(Vector4 *res, Vector4 *a, Vector4 *b)
{
    asm volatile ("mov %0, %%eax"::"m"(a));
    asm volatile ("mov %0, %%ebx"::"m"(b));
    asm volatile ("movups (%eax), %xmm0");
    asm volatile ("movups (%ebx), %xmm1");
    asm volatile ("addps %xmm1, %xmm0");
    asm volatile ("mov %0, %%eax"::"m"(res));
    asm volatile ("movups %xmm0, (%eax)");
}

void MatrixSSE::SSE_Sub(Vector4 *res, Vector4 *a, Vector4 *b)
{
    asm volatile ("mov %0, %%eax"::"m"(a));
    asm volatile ("mov %0, %%ebx"::"m"(b));
    asm volatile ("movups (%eax), %xmm0");
    asm volatile ("movups (%ebx), %xmm1");
    asm volatile ("subps %xmm1, %xmm0");
    asm volatile ("mov %0, %%eax"::"m"(res));
    asm volatile ("movups %xmm0, (%eax)");
}

void MatrixSSE::SSE_Mul(Vector4 *res, Vector4 *a, Vector4 *b)
{
    asm volatile ("mov %0, %%eax"::"m"(a));
    asm volatile ("mov %0, %%ebx"::"m"(b));
    asm volatile ("movups (%eax), %xmm0");
    asm volatile ("movups (%ebx), %xmm1");
    asm volatile ("mulps %xmm1, %xmm0");
    asm volatile ("mov %0, %%eax"::"m"(res));
    asm volatile ("movups %xmm0, (%eax)");
}

void MatrixSSE::allocSpace()
{
    p = new float[N_ * N_];
}

MatrixSSE operator+(const MatrixSSE& m1, const MatrixSSE& m2)
{
    MatrixSSE temp(m1);
    return (temp += m2);
}

MatrixSSE operator-(const MatrixSSE& m1, const MatrixSSE& m2)
{
    MatrixSSE temp(m1);
    return (temp -= m2);
}

MatrixSSE operator*(const MatrixSSE& m1, const MatrixSSE& m2)
{
    MatrixSSE temp(m1);
    return (temp *= m2);
}

MatrixSSE operator*(const MatrixSSE& m, float num)
{
    MatrixSSE temp(m);
    return (temp *= num);
}

MatrixSSE operator*(float num, const MatrixSSE& m)
{
    return (m * num);
}

MatrixSSE operator/(const MatrixSSE& m, float num)
{
    MatrixSSE temp(m);
    return (temp /= num);
}

std::ostream& operator<<(std::ostream& os, const MatrixSSE& m)
{
    for (int i = 0; i < m.N_; ++i) {
        os << MatrixSSE::round_( m.p[i * m.N_], 3);
        for (int j = 1; j < m.N_; ++j) {
            os << " " << MatrixSSE::round_( m.p[i * m.N_ + j], 3);
        }
        os << std::endl;
    }
    return os;
}

std::istream& operator>>(std::istream& is, MatrixSSE& m)
{
    for (int i = 0; i < m.N_; ++i) {
        for (int j = 0; j < m.N_; ++j) {
            is >> m.p[i * m.N_ + j];
        }
    }
    return is;
}

float MatrixSSE::round_(float x, int pr) {
    float d = pow(10, pr);
    return round(x * d ) / d + 0.0;
}
