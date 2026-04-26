# Лабораторная работа №2

**Студент:** Стребелев Семён Алексеевич

**Группа:** 6311-100503D

---

## 1 Цель работы

Модифицировать программу умножения квадратных матриц из лабораторной работы №1 для параллельной работы 1по технологии CUDA. Провести эксперименты с разными размерами матриц и различными конфигурациями сетки блоков.

---

## 2. Теоретические сведения

CUDA — это технология NVIDIA для параллельных вычислений на графическом процессоре. Она позволяет переносить часть вычислений с центрального процессора на видеокарту и за счет этого ускорять выполнение программы.

В данной работе CUDA используется для параллельного умножения матриц. Вычисления выполняются на GPU, а эффективность программы зависит от размера входных данных и выбранной конфигурации блоков.

---

## 3. Описание алгоритма

1. Считать матрицу $A$ из файла `A.txt`
2. Считать матрицу $B$ из файла `B.txt`
3. Проверить совпадение размерностей матриц
4. Преобразовать двумерные матрицы в одномерные массивы
5. Выделить память на GPU
6. Скопировать данные матриц $A$ и $B$ из оперативной памяти на GPU
7. Запустить CUDA-ядро для вычисления произведения матриц
8. Выполнить измерение времени работы программы при разных конфигурациях блоков
9. Скопировать результирующую матрицу из памяти GPU обратно в оперативную память
10. Сохранить результат в файл `RESULT.txt`
11. Вывести в консоль.
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
- размер блока
- время выполнения
- количество операций

Пример для матрицы 3x3:

```
N = 3
Block = 8 x 8
Time = 0.361472 ms
Operations = 45
```


---

## 8 Характеристики системы

- Процессор: AMD Ryzen 5 5600H with Radeon Graphics

- Количество ядер: 6 физических ядер / 12 логических потоков

- ОЗУ: 16 ГБ 

- ОС: Windows
  
- Видеокарта: NVIDIA GeForce RTX 3050 Laptop GPU


---
## 9 Исследование программы

Для исследования зависимости времени выполнения от размера задачи были проведены эксперименты с различными размерами матриц.

| Размер N | Block 8×8 | Block 16×16 | Block 32×32 |
|:--------:|:---------:|:-----------:|:-----------:|
| 100 | 0.000480128 | 0.000226016 | 0.00022 |
| 200 | 0.0011935 | 0.0009304 | 0.00132957 |
| 400 | 0.00662419 | 0.00616182 | 0.00804656 |
| 800 | 0.0522408 | 0.0421442 | 0.061037 |
| 1200 | 0.167037 | 0.0916332 | 0.133063 |
| 1600 | 0.303072 | 0.214302 | 0.313872 |
| 2000 | 0.534502 | 0.412468 | 0.611591 |


---

## 10 График зависимости времени

<img width="1212" height="736" alt="image" src="https://github.com/user-attachments/assets/6c3773cf-9b40-4a36-a974-b21ee2ac06d8" />



---

## 11 Анализ результатов

Для исследования зависимости времени выполнения от размера задачи программа запускалась на матрицах разных размеров.

В ходе экспериментов можно использовать размеры:

- 100x100
- 200x200
- 400x400
- 600х600
- 800x800
- 1200x1200
- 1600x1600
- 2000x2000

и размер блоков:
- 8x8
- 16x16
- 32x32

По результатам экспериментов видно, что при увеличении размера матрицы время выполнения растет.

На большинстве тестов лучшей оказалась конфигурация 16x16. Она показала минимальное время для размеров 200, 400, 800, 1200, 1600 и 2000.
Блок 8x8 во всех случаях работает медленнее.
Блок 32x32 на малых размерах дает близкий результат, но на больших матрицах тоже уступает 16x16.

Таким образом, для данной реализации и данной видеокарты наиболее эффективной оказалась конфигурация 16x16.

---

## 12 Вывод

В ходе лабораторной работы была модифицирована программа умножения квадратных матриц для параллельного выполнения с использованием технологии CUDA. Были проведены эксперименты с различными размерами матриц и разными конфигурациями блоков потоков.

Эксперименты показали, что при увеличении размера матрицы время выполнения возрастает, а выбор конфигурации блоков существенно влияет на производительность программы. Наилучшие результаты в большинстве запусков показала конфигурация 16x16, которая оказалась наиболее эффективной для используемой видеокарты NVIDIA GeForce RTX 3050 Laptop GPU.

Таким образом, применение CUDA позволило эффективно распараллелить вычисления, а исследование подтвердило, что правильный выбор размера блока является важным фактором повышения производительности программы

---
