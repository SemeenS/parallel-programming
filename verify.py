import numpy as np

def read_matrix(filename):
    with open(filename) as f:
        n = int(f.readline())
        data = []
        for _ in range(n):
            data.append(list(map(float, f.readline().split())))
        return np.array(data)

A = read_matrix("A.txt")
B = read_matrix("B.txt")
C = read_matrix("RESULT.txt")

C_expected = np.dot(A, B)

if np.allclose(C, C_expected, atol=1e-10):
    print("Verification passed!")
    print(f"Max difference: {np.max(np.abs(C - C_expected)):.2e}")
else:
    print("Verification failed: Results do not match!")
    print(f"Max difference: {np.max(np.abs(C - C_expected)):.2e}")