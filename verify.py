import numpy as np

def read_matrix(filename):
    with open(filename, "r") as f:
        n = int(f.readline())
        matrix = []

        for i in range(n):
            row = list(map(float, f.readline().split()))
            matrix.append(row)

    return np.array(matrix)

A = read_matrix("A.txt")
B = read_matrix("B.txt")
C = read_matrix("RESULT.txt")

if np.allclose(C, A @ B):
    print("Verification passed")
else:
    print("Verification failed")