#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "File open error: " << filename << "\n";
        exit(1);
    }

    file >> n;

    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> matrix[i][j];
        }
    }

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix, int n) {
    ofstream file(filename);

    if (!file.is_open()) {
        cout << "File create error: " << filename << "\n";
        exit(1);
    }

    file << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j] << " ";
        }
        file << "\n";
    }
}

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int n1, n2;

    vector<vector<double>> A = readMatrix("pp_lab1/A.txt", n1);
    vector<vector<double>> B = readMatrix("pp_lab1/B.txt", n2);

    if (n1 != n2) {
        cout << "Matrix sizes are different\n";
        return 1;
    }

    int N = n1;

    int blockSize = 16; // меняй на 8, 16, 32 для экспериментов

    vector<double> h_A(N * N), h_B(N * N), h_C(N * N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = A[i][j];
            h_B[i * N + j] = B[i][j];
        }
    }

    double* d_A;
    double* d_B;
    double* d_C;

    size_t bytes = (size_t)N * N * sizeof(double);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(blockSize, blockSize);
    dim3 blocks((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulKernel << <blocks, threads >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeMs = 0.0f;
    cudaEventElapsedTime(&timeMs, start, stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    vector<vector<double>> C(N, vector<double>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = h_C[i * N + j];
        }
    }

    writeMatrix("pp_lab1/RESULT.txt", C, N);

    long long mult = 1LL * N * N * N;
    long long add = 1LL * N * N * (N - 1);
    long long ops = mult + add;

    cout << "N = " << N << "\n";
    cout << "Block = " << blockSize << " x " << blockSize << "\n";
    cout << "Time = " << timeMs << " ms\n";
    cout << "Operations = " << ops << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}