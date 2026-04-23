import random

n = 10

def write_matrix(filename, n):
    with open(filename, "w") as f:
        f.write(str(n) + "\n")
        for i in range(n):
            row = [str(random.randint(0, 9)) for _ in range(n)]
            f.write(" ".join(row) + "\n")

write_matrix("A.txt", n)
write_matrix("B.txt", n)

print(f"Generated {n}x{n}")