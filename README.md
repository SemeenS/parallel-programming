# Лабораторная работа №1
## Перемножение двух квадратных матриц

**Выполнил студент:** Стребелев Семен  
**Группа:** 6311-100503D

---

## 1. Цель работы

Реализовать алгоритм перемножения квадратных матриц, измерить время выполнения программы, сравнить с результатом библиотеки NumPy.

---

## 2. Теоретические сведения

Произведение двух квадратных матриц определяется по формуле:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

где:
- A — первая матрица
- B — вторая матрица
- C — результирующая матрица

При стандартном алгоритме перемножения квадратных матриц размером \(n \times n\):
- число умножений: $$(n^3)$$
- число сложений: $$(n^2 (n - 1))$$
- общее количество арифметических операций:

$$Operations = 2N^3$$

Алгоритмическая сложность программы:

$$O(N^3)$$

---

## 3. Описание алгоритма

Алгоритм работы программы:
1. Сгенерировать входные данные.
2. Считать матрицу $A$ из файла `A.txt`
3. Считать матрицу $B$ из файла `B.txt`
4. Проверить совпадение размерностей
5. Создать результирующую матрицу $C$
6. Выполнить перемножение матриц с помощью трёх вложенных циклов
7. Измерить время выполнения
8. Сохранить результат в файл `RESULT.txt`
9. Вывести размер матрицы, время выполнения и объём вычислений

---

## 4 Автоматическая генерация матриц

Для генерации матриц используется Python

```python
import random

n = 3

def write_matrix(filename, n):
    with open(filename, "w") as f:
        f.write(str(n) + "\n")
        for i in range(n):
            row = [str(random.randint(0, 9)) for _ in range(n)]
            f.write(" ".join(row) + "\n")

write_matrix("A.txt", n)
write_matrix("B.txt", n)

print(f"Generated {n}x{n}")
```

---

## 5. Исходный код программы

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

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

int main() {
    int n1, n2;

    vector<vector<double>> A = readMatrix("pp_lab1/A.txt", n1);
    vector<vector<double>> B = readMatrix("pp_lab1/B.txt", n2);

    if (n1 != n2) {
        cout << "Matrix sizes are different\n";
        return 1;
    }

    int n = n1;
    vector<vector<double>> C(n, vector<double>(n, 0));

    auto start = chrono::high_resolution_clock::now();

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
    cout << "Time = " << time_sec << " sec\n";
    cout << "Operations = " << ops << "\n";

    return 0;
}
```
---

## 6 Формат входных данных

A.txt:
```
3
9 8 8
3 2 7
3 1 1

```

B.txt:
```
3
6 2 3
3 6 6
6 1 2


```

---

## 7 Формат выходных данных

RESULT.txt
```
3
126 74 91 
66 25 35 
27 13 17 


```

Также в консоль выводятся: 
- размер матрицы
- время выполнения
- количество операций

Пример для матрицы 3x3:

```
N = 3
Time = 1.5e-06 sec
Operations = 45
```

---

## 8 Автоматическая верификация результатов

Для проверки корректности используется Python и библиотека NumPy.

```python
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
```

**Результат выполнения:**
```
Verification passed!
Max difference: 0.00e+00
```

---

## 9 Исследование программы

Для исследования зависимости времени выполнения от размера задачи были проведены эксперименты с различными размерами матриц.

| Размер матрицы N  | Время выполнения(сек) | Количество операций  |
| --- | :---: | ---: |
| 100 | 0.0452226 | 1990000 |
| 200 | 0.328577 | 15960000 |
| 400 | 2.4552 | 127840000 |
| 800 | 15.9219 | 1023360000 |
| 1200 | 48.3915 | 3454560000 |
| 1600 | 119.406 | 8189440000 |
| 2000 | 243.757| 15996000000 |

---

## 10 График зависимости времени

<img width="1164" height="717" alt="image" src="https://github.com/user-attachments/assets/9521e2f4-1047-44a2-bac3-3b1f5e02ee47" />

---

## 11 Анализ результатов

Для исследования зависимости времени выполнения от размера задачи программа запускалась на матрицах разных размеров.

В ходе экспериментов можно использовать размеры:

- 100x100
- 200x200
- 400x400
- 800x800
- 1200x1200
- 1600x1600
- 2000x2000

При увеличении размера матрицы время выполнения возрастает, что соответствует теоретической сложности $$O(N^3)$$

---

## 12 Вывод

В ходе лабораторной работы была реализована программа на языке C++ для перемножения двух квадратных матриц с чтением данных из файлов и записью результата в файл. Также была выполнена автоматическая верификация результата с помощью Python и библиотеки NumPy. Исследования показали, что с ростом размерности матриц время выполнения увеличивается.

---
