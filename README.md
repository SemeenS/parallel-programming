# Лабораторная работа №3

**Студент:** Стребелев Семён Алексеевич  
**Группа:** 6311-100503D

## 1. Цель работы

Модифицировать программу из лабораторной работы №1 для параллельной работы с использованием технологии **MPI**.  
Провести исследование зависимости времени выполнения программы от размера матриц и числа используемых вычислительных ядер (MPI-процессов), а также проанализировать масштабируемость программы.

---

## 2. Теоретические сведения

### 2.1. MPI (Message Passing Interface)

MPI — это технология параллельного программирования, основанная на модели распределённой памяти.  
В MPI каждый процесс имеет собственную память, а обмен данными между процессами выполняется через передачу сообщений.

MPI широко применяется для построения параллельных программ на многопроцессорных системах и вычислительных кластерах.

### 2.2. Основные функции MPI, использованные в работе

| Функция | Назначение |
|---|---|
| `MPI_Init` | инициализация MPI |
| `MPI_Comm_rank` | получение номера текущего процесса |
| `MPI_Comm_size` | получение общего числа процессов |
| `MPI_Bcast` | рассылка данных всем процессам |
| `MPI_Scatterv` | распределение частей массива между процессами |
| `MPI_Gatherv` | сбор частей результата от всех процессов |
| `MPI_Barrier` | синхронизация процессов |
| `MPI_Wtime` | измерение времени выполнения |
| `MPI_Reduce` | свёртка данных от процессов |
| `MPI_Finalize` | завершение работы MPI |

### 2.3. Метрики производительности

В работе используются следующие показатели:

**Ускорение:**

$S_p = \frac{T_1}{T_p}$

где:
- $T_1$ — время выполнения на одном процессе,
- $T_p$ — время выполнения на $p$ процессах.

**Эффективность:**

$E_p = \frac{S_p}{p}$

**Число арифметических операций** при умножении двух квадратных матриц размера $N \times N$:

$Ops = N^3 + N^2(N - 1)$

Алгоритмическая сложность программы:

$$O(N^3)$$


---

## 3. Описание алгоритма

Программа выполняет умножение двух квадратных матриц, считанных из файлов `A.txt` и `B.txt`, и записывает результат в файл `RESULT.txt`.

### 3.1. Общая схема работы

1. Главный процесс (`rank`) считывает матрицы из файлов.
2. Проверяется совпадение размеров матриц.
3. Матрицы преобразуются в одномерный вид для удобной передачи через MPI.
4. Размер матрицы `n` рассылается всем процессам.
5. Матрица `A` делится между процессами по строкам.
6. Матрица `B` целиком рассылается всем процессам.
7. Каждый процесс вычисляет свою часть результирующей матрицы `C`.
8. Главный процесс собирает все части результата.
9. Главный процесс записывает итоговую матрицу в файл `RESULT.txt`.

### 3.2. Распределение данных

При распараллеливании использован следующий подход:

- матрица **A** разбивается по строкам между процессами;
- матрица **B** передаётся всем процессам полностью;
- каждый процесс вычисляет только свой набор строк результирующей матрицы;
- результат собирается обратно на главном процессе с помощью `MPI_Gatherv`.

При таком подходе каждая строка матрицы `C` может вычисляться независимо.

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

        vector<vector<double>> A = readMatrix("pp_lab1/A.txt", n1);
        vector<vector<double>> B = readMatrix("pp_lab1/B.txt", n2);

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
        writeMatrix("pp_lab1/RESULT.txt", C, n);

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
- количество процессов
- время выполнения
- количество операций

Пример для матрицы 3x3:

```
N = 3
Processes = 1
Time = 2.55e-05 sec
Operations = 45
```

---

## 8 Характеристики системы

- Процессор: AMD Ryzen 5 5600H with Radeon Graphics

- Количество ядер: 6 физических ядер / 12 логических потоков

- ОЗУ: 16 ГБ 

- ОС: Windows

---

## 9 Исследование программы

Для исследования зависимости времени выполнения от размера задачи были проведены эксперименты с различными размерами матриц.

| Размер матрицы N  | 1 процесс | 2 процесс | 4 процесс | 8 процесс |
| ---: | ---: | ---: | ---: | ---: |
| 100 |0.0052034|0.0026967|0.0029273|0.0019142|
| 200 |0.041563|0.0212429|0.0110415|0.0110181|
| 400 |0.33858|0.17606|0.104411|0.0795983|
| 800 |2.7479|1.38995|0.866978|0.659786|
| 1200 |9.45583|4.8845|2.75837|2.09621|
| 1600 |23.4921|16.7596|9.14523|5.88834|
| 2000 |48.1603|23.7701|13.5831|10.5042|

---
## 10 График зависимости времени

<img width="1883" height="923" alt="image" src="https://github.com/user-attachments/assets/669cc580-5ade-45dc-b009-bdad03093103" />

<img width="1222" height="725" alt="image" src="https://github.com/user-attachments/assets/392cf50b-f8af-495f-8687-31697c6b1c0c" />

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

и число используемых вычислительных ядер:
- 1
- 2
- 4
- 8

По результатам экспериментов можно сделать следующие выводы:

- при увеличении размера матрицы время выполнения возрастает, что соответствует теоретической сложности алгоритма $O(N^3)$;
- при увеличении числа MPI-процессов время выполнения в целом уменьшается;
- наиболее заметный эффект распараллеливания наблюдается на больших размерах матриц, где вычислительная нагрузка становится существенно выше;
- для небольших матриц выигрыш от увеличения числа процессов меньше, так как заметную роль начинают играть накладные расходы на создание процессов, обмен данными и синхронизацию;
- при переходе от 1 процесса к 2, 4 и 8 процессам наблюдается устойчивое сокращение времени выполнения;
- ускорение не является строго линейным, поскольку на эффективность MPI-варианта влияют затраты на передачу данных между процессами, синхронизацию и особенности аппаратной платформы;
- для матриц большого размера, например 1600x1600 и 2000x2000, использование 8 процессов даёт наибольший выигрыш по времени по сравнению с однопроцессным запуском.
- на всех общих размерах матриц MPI показал лучшее время выполнения, чем OpenMP, в проведённых экспериментах преимущество MPI составило примерно от 3.18 до 4.56 раза;

В рамках данной лабораторной работы реализация на MPI показала более высокую производительность по сравнению с реализацией на OpenMP.

Таким образом, результаты экспериментов показывают, что применение MPI позволяет эффективно распараллелить задачу умножения квадратных матриц, особенно при больших размерах входных данных.

---

## 12 Вывод

В ходе лабораторной работы была модифицирована программа последовательного умножения квадратных матриц для параллельного выполнения с использованием MPI.
В программе была реализована передача данных между процессами, распределение строк матрицы между MPI-процессами, параллельное вычисление части результирующей матрицы каждым процессом и последующий сбор результата на главном процессе.
Также было проведено исследование зависимости времени выполнения программы от размера матриц и количества MPI-процессов.
Результаты экспериментов показывают, что применение MPI позволяет уменьшить время выполнения программы, особенно при работе с матрицами большого размера. При этом наиболее заметный эффект достигается при увеличении числа процессов на задачах с высокой вычислительной сложностью.
Таким образом, технология MPI является эффективным средством распараллеливания задачи умножения матриц и позволяет существенно повысить производительность программы по сравнению с последовательным вариантом.

---
