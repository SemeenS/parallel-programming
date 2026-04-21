#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

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

int main(int argc, char* argv[]) {
    int n1, n2;
    int threads = 8;

    if (argc > 1) {
        threads = atoi(argv[1]);
        if (threads <= 0) {
            cout << "Invalid thread count\n";
            return 1;
        }
    }

    vector<vector<double>> A = readMatrix("pp_lab1/A.txt", n1);
    vector<vector<double>> B = readMatrix("pp_lab1/B.txt", n2);

    if (n1 != n2) {
        cout << "Matrix sizes are different\n";
        return 1;
    }

    int n = n1;
    vector<vector<double>> C(n, vector<double>(n, 0));

    omp_set_num_threads(threads);

    auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();

    writeMatrix("pp_lab1/RESULT.txt", C, n);

    double time_sec = chrono::duration<double>(end - start).count();

    long long mult = 1LL * n * n * n;
    long long add = 1LL * n * n * (n - 1);
    long long ops = mult + add;

    cout << "N = " << n << "\n";
    cout << "Threads = " << threads << "\n";
    cout << "Time = " << time_sec << " sec\n";
    cout << "Operations = " << ops << "\n";

    return 0;
}