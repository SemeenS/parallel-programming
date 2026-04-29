# Лабораторная работа №5
## Параллельное перемножение матриц на суперкомпьютере

**Студент:** Стребелев Семён Алексеевич


**Группа:** 6311-100503D

---

## 1 Цель работы

Модифицировать программу умножения квадратных матриц для параллельной работы с использованием технологии MPI.  
Запустить параллельную версию программы на суперкомпьютере «Сергей Королёв» и провести эксперименты с различными размерами матриц и различным количеством MPI-процессов.

---

## 2 Теоретические сведения

Произведение матриц определяется формулой:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

где:
- A — первая матрица,
- B — вторая матрица,
- C — результирующая матрица.

Количество операций при умножении квадратных матриц размерности N:

$$Operations = 2N^3$$

Алгоритмическая сложность:

$$O(N^3)$$

---
## 3 Описание параллельного алгоритма

Алгоритм работы MPI-программы:

1. Главный процесс считывает матрицу `A` из файла `A.txt`.
2. Главный процесс считывает матрицу `B` из файла `B.txt`.
3. Проверяется совпадение размерностей матриц.
4. Двумерные матрицы преобразуются в одномерные массивы.
5. Размер матриц рассылается всем MPI-процессам.
6. Матрица `B` полностью рассылается всем процессам.
7. Строки матрицы `A` распределяются между процессами.
8. Каждый процесс вычисляет свою часть результирующей матрицы `C`.
9. Части матрицы `C` собираются на главном процессе.
10. Главный процесс записывает результат в файл `RESULT.txt`.
11. В консоль выводятся размер матрицы, количество процессов, время выполнения и количество операций.
---
## 4 Исходный код программы (MPI)

```cpp
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstddef>

using namespace std;

vector<vector<double> > readMatrix(const string& filename, int& n) {
    ifstream file(filename.c_str());

    if (!file.is_open()) {
        cout << "File open error: " << filename << "\n";
        exit(1);
    }

    file >> n;

    vector<vector<double> > matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> matrix[i][j];
        }
    }

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double> >& matrix, int n) {
    ofstream file(filename.c_str());

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

vector<double> flattenMatrix(const vector<vector<double> >& matrix, int n) {
    vector<double> flat(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat[i * n + j] = matrix[i][j];
        }
    }

    return flat;
}

vector<vector<double> > unflattenMatrix(const vector<double>& flat, int n) {
    vector<vector<double> > matrix(n, vector<double>(n));

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

        vector<vector<double> > A = readMatrix("A.txt", n1);
        vector<vector<double> > B = readMatrix("B.txt", n2);

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
        rank == 0 ? &Aflat[0] : NULL,
        &sendCounts[0],
        &displs[0],
        MPI_DOUBLE,
        localA.empty() ? NULL : &localA[0],
        localRows * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(&Bflat[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < localRows; i++) {
        for (int k = 0; k < n; k++) {
            double a = localA[i * n + k];
            for (int j = 0; j < n; j++) {
                localC[i * n + j] += a * Bflat[k * n + j];
            }
        }
    }

    MPI_Gatherv(
        localC.empty() ? NULL : &localC[0],
        localRows * n,
        MPI_DOUBLE,
        rank == 0 ? &Cflat[0] : NULL,
        &sendCounts[0],
        &displs[0],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();
    double localTime = end - start;
    double maxTime = 0.0;

    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<vector<double> > C = unflattenMatrix(Cflat, n);
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
```
---
## 5 Формат входных данных

matrix_1.txt:
B.txt
```
3
9 8 8
3 2 7
3 1 1
```

matrix_2.txt:
A.txt
```
3
6 2 3
3 6 6
6 1 2
```
---
## 6 Формат выходных данных
RESULT.txt
```
3
126 74 91 
66 25 35 
27 13 17 
```
---
### 7 Пример вывода для умножения матриц 10х10 с 8 процессами:


```
N = 10
Processes = 8
Time = 9.799e-05 sec
Operations = 1900


```

---

### 8 Результаты эксперемента

| Размер N | 1 процесс | 2 процесса | 4 процесса | 8 процессов |
|----------|-----------|------------|------------|-------------|
| 200      | 0.084038   | 0.0423031  | 0.01249    | 0.011168   |
| 400      | 0.667224    | 0.334297   | 0.171503  | 0.0859962   |
| 800      | 4.92878    | 2.58901    | 1.35824   | 0.685866     |
| 1200     | 16.1139    | 8.42298   | 4.25643    | 2.22473     |
| 1600     | 39.0617    | 19.4766   | 9.8749    | 5.08096     |
| 2000     | 75.3078    | 37.6047   | 19.2484    | 9.7779     |

---


## 9 Графики

<img width="1174" height="739" alt="image" src="https://github.com/user-attachments/assets/333f1998-faaf-4ed9-ad65-92443986e2df" />

---

## 10 Анализ результатов

Для исследования зависимости времени выполнения от размера задачи программа запускалась на матрицах разных размеров:

* 200x200;
* 400x400;
* 800x800;
* 1200x1200;
* 1600x1600;
* 2000x2000.

Для каждого размера матрицы были выполнены запуски с разным количеством MPI-процессов:

* 1 процесс;
* 2 процесса;
* 4 процесса;
* 8 процессов.

По результатам экспериментов видно, что при увеличении размера матрицы время выполнения возрастает. Это связано с тем, что сложность классического алгоритма умножения матриц составляет `O(N^3)`.
Также видно, что увеличение количества MPI-процессов уменьшает время выполнения программы. 
На больших размерах матриц ускорение проявляется лучше, так как вычислительная нагрузка становится достаточно большой и затраты на обмен данными между процессами меньше влияют на общее время выполнения. На малых размерах матриц разница между 4 и 8 процессами менее заметна, так как накладные расходы на распределение и сбор данных становятся более существенными.

Таким образом, для данной реализации наилучшее время выполнения было получено при запуске на 8 MPI-процессах.

---

## 11 Вывод

В ходе лабораторной работы была модифицирована программа умножения квадратных матриц для параллельного выполнения с использованием технологии MPI. Программа была запущена на суперкомпьютере «Сергей Королёв».

Были проведены эксперименты с различными размерами матриц и различным количеством MPI-процессов. Результаты показали, что при увеличении размера матрицы время выполнения возрастает, а при увеличении количества MPI-процессов время выполнения уменьшается.

Наиболее эффективный результат в проведенных экспериментах был получен при запуске программы на 8 процессах. Это подтверждает, что использование MPI позволяет эффективно распараллелить вычисления при умножении больших квадратных матриц.


---
