//
// Created by kabanpunk on 25.11.2021.
//

#ifndef LAB7_MATRIXSSE_H
#define LAB7_MATRIXSSE_H

#include <iostream>

class MatrixSSE {
    typedef struct { float x, y, z, w; } Vector4;

public:
    explicit MatrixSSE(int);
    MatrixSSE(int, int, int);
    MatrixSSE(float*, int);
    MatrixSSE();
    ~MatrixSSE();
    MatrixSSE(const MatrixSSE&);
    MatrixSSE& operator=(const MatrixSSE&);

    inline float& operator()(int x, int y) { return p[x * N_ + y]; }

    MatrixSSE& operator+=(const MatrixSSE&);
    MatrixSSE& operator-=(const MatrixSSE&);
    MatrixSSE& operator*=(const MatrixSSE&);
    MatrixSSE& operator*=(float);
    MatrixSSE& operator/=(float);

    friend std::ostream& operator<<(std::ostream&, const MatrixSSE&);
    friend std::istream& operator>>(std::istream&, MatrixSSE&);

    MatrixSSE transpose();
    float getMaxRowsSum();
    float getMaxCollumnsSum();

    static MatrixSSE createIdentity(int);
    static void SSE_Add(Vector4*, Vector4*, Vector4*);
    static void SSE_Sub(Vector4*, Vector4*, Vector4*);
    static void SSE_Mul(Vector4*, Vector4*, Vector4*);
    static Vector4 setVector(const MatrixSSE&, int);

private:
    int N_;
    int NS;
    float *p;

    static float round_(float, int);
    void allocSpace();
};

MatrixSSE operator+(const MatrixSSE&, const MatrixSSE&);
MatrixSSE operator-(const MatrixSSE&, const MatrixSSE&);
MatrixSSE operator*(const MatrixSSE&, const MatrixSSE&);
MatrixSSE operator*(const MatrixSSE&, float);
MatrixSSE operator*(float, const MatrixSSE&);
MatrixSSE operator/(const MatrixSSE&, float);

#endif //LAB7_MATRIXSSE_H
