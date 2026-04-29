#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

vector<double> flattenMatrix(const vector<vector<double>>& matrix, int n) {
    vector<double> flat(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat[i * n + j] = matrix[i][j];
        }
    }

    return flat;
}

vector<vector<double>> unflattenMatrix(const vector<double>& flat, int n) {
    vector<vector<double>> matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = flat[i * n + j];
        }
    }

    return matrix;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    int n = 0;
    vector<double> Aflat;
    vector<double> Bflat;
    vector<double> Cflat;

    if (rank == 0) {
        int n1 = 0;
        int n2 = 0;

        vector<vector<double>> A = readMatrix("A.txt", n1);
        vector<vector<double>> B = readMatrix("B.txt", n2);

        if (n1 != n2) {
            cout << "Matrix sizes are different\n";
            n = -1;
        }
        else {
            n = n1;
            Aflat = flattenMatrix(A, n);
            Bflat = flattenMatrix(B, n);
            Cflat.resize(n * n);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        MPI_Finalize();
        return 1;
    }

    if (rank != 0) {
        Bflat.resize(n * n);
    }

    vector<int> rowsPerProcess(processCount);
    vector<int> sendCounts(processCount);
    vector<int> displs(processCount);

    int baseRows = n / processCount;
    int extraRows = n % processCount;
    int offset = 0;

    for (int p = 0; p < processCount; p++) {
        rowsPerProcess[p] = baseRows + (p < extraRows ? 1 : 0);
        sendCounts[p] = rowsPerProcess[p] * n;
        displs[p] = offset;
        offset += sendCounts[p];
    }

    int localRows = rowsPerProcess[rank];
    vector<double> localA(localRows * n);
    vector<double> localC(localRows * n, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(
        rank == 0 ? Aflat.data() : nullptr,
        sendCounts.data(),
        displs.data(),
        MPI_DOUBLE,
        localA.empty() ? nullptr : localA.data(),
        localRows * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(Bflat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < localRows; i++) {
        for (int k = 0; k < n; k++) {
            double a = localA[i * n + k];
            for (int j = 0; j < n; j++) {
                localC[i * n + j] += a * Bflat[k * n + j];
            }
        }
    }

    MPI_Gatherv(
        localC.empty() ? nullptr : localC.data(),
        localRows * n,
        MPI_DOUBLE,
        rank == 0 ? Cflat.data() : nullptr,
        sendCounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();
    double localTime = end - start;
    double maxTime = 0.0;

    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<vector<double>> C = unflattenMatrix(Cflat, n);
        writeMatrix("RESULT.txt", C, n);

        long long mult = 1LL * n * n * n;
        long long add = 1LL * n * n * (n - 1);
        long long ops = mult + add;

        cout << "N = " << n << "\n";
        cout << "Processes = " << processCount << "\n";
        cout << "Time = " << maxTime << " sec\n";
        cout << "Operations = " << ops << "\n";
    }

    MPI_Finalize();
    return 0;
}