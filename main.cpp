#include <iostream>
#include "matrixSSE.h"
#include "matrix.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <chrono>
using namespace std::chrono;

void TEST1()
{
    int N = 3, M = 100000;
    Matrix A(N, 0, 10);
    Matrix R(N);
    Matrix B(N);
    Matrix I(N);
    I = Matrix::createIdentity(N);

    B = A.transpose() / (A.getMaxRowsSum() * A.getMaxCollumnsSum());
    R = I - (B * A);

    Matrix R_(N);
    Matrix RS(N);
    R_ = R;
    for (int i = 0; i < M; ++i)
    {
        RS += R_;
        R_ *= R;
    }

    Matrix AI(N);
    AI = (I + RS) * B;

    //std::cout << A << std::endl;
    //std::cout << AI << std::endl;
    //std::cout << AI * A << std::endl;
}

void TEST2()
{
    int N = 3, M = 100000;
    MatrixSSE A(N, 0, 10);
    MatrixSSE R(N);
    MatrixSSE B(N);
    MatrixSSE I(N);
    I = MatrixSSE::createIdentity(N);

    B = A.transpose() / (A.getMaxRowsSum() * A.getMaxCollumnsSum());
    R = I - (B * A);

    MatrixSSE R_(N);
    MatrixSSE RS(N);
    R_ = R;
    for (int i = 0; i < M; ++i)
    {
        RS += R_;
        R_ *= R;
    }

    MatrixSSE AI(N);
    AI = (I + RS) * B;

    //std::cout << A << std::endl;
    //std::cout << AI << std::endl;
    //std::cout << AI * A << std::endl;
}

void TEST3()
{
    int N = 3, M = 100000;
    gsl_matrix* A = gsl_matrix_alloc(N, N);

    int a = 0, b = 10;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A->data[i * N + j] = rand() % (b - a) + a;
        }
    }

    gsl_matrix* R = gsl_matrix_alloc(N, N);
    gsl_matrix* B = gsl_matrix_alloc(N, N);
    gsl_matrix* I = gsl_matrix_alloc(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j)
                I->data[i * N + j] = 1;
            else
                I->data[i * N + j] = 0;
        }
    }

    double maxRowsSum = -1;
    double maxCollumnsSum = -1;

    for (int i = 0; i < N; ++i) {
        double sr = 0;
        double sc = 0;
        for (int j = 0; j < N; ++j) {
            sr += fabs(A->data[i * N + j]);
            sr += fabs(A->data[j * N + i]);
        }
        maxRowsSum = std::max(maxRowsSum, sr);
        maxCollumnsSum = std::max(maxCollumnsSum, sr);
    }

    gsl_matrix* AT = gsl_matrix_alloc(N, N);
    gsl_matrix_memcpy(AT, A);
    gsl_matrix_transpose(AT);

    // C = alpha * A * B + beta * C
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1. / (maxRowsSum * maxCollumnsSum), AT, I, 0., B);

    gsl_matrix_memcpy(R, I);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1., B, A, 1., R); // R = -1 * B * A + 1 * R

    gsl_matrix* R_ = gsl_matrix_alloc(N, N);
    gsl_matrix* RS = gsl_matrix_alloc(N, N);
    gsl_matrix_memcpy(R_, R);
    for (int i = 0; i < M; ++i) {
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., R_, I, 1., RS); //RS += R_ ~ RS = 1 * R_ * I + 1 * RS
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., R_, R, 0., R_); //R_ *= R; ~ R_ = 1 * R_ * R + 0 * R_
    }

    // AI = (I + RS) * B
    gsl_matrix* AI = gsl_matrix_alloc(N, N);
    gsl_matrix* RSI = gsl_matrix_alloc(N, N);
    gsl_matrix_memcpy(RSI, I);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., RS, I, 1., RSI); // RSI = 1 * RS * I + 1 * I
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., RSI, B, 0., AI);  // AI = RSI * B
}

int main() {
    time_point start = high_resolution_clock::now();
    TEST1();
    time_point stop = high_resolution_clock::now();
    duration duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1000000. << " sec." << std::endl << std::endl;

    start = high_resolution_clock::now();
    TEST2();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1000000. << " sec." << std::endl << std::endl;

    start = high_resolution_clock::now();
    TEST3();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1000000. << " sec." << std::endl << std::endl;

    return 0;
}
