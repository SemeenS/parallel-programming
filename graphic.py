import matplotlib.pyplot as plt
import numpy as np

# Данные из таблицы
matrix_sizes = [100, 200, 400, 600, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 6, 8, 10, 12]

times = {
    1:  [0.0229074, 0.180585, 1.44213, 4.80928, 11.5009, 41.121, 103.048, 193.807],
    2:  [0.0122327, 0.0908257, 0.726143, 2.46056, 6.00807, 26.2365, 58.9564, 139.754],
    4:  [0.0069199, 0.0492419, 0.472217, 1.37356, 3.35089, 13.3445, 29.5304, 82.638],
    6:  [0.0072088, 0.0365236, 0.443567, 1.03029, 2.64271, 10.374, 24.1306, 51.3074],
    8:  [0.0070624, 0.0457279, 0.349114, 1.04644, 2.6799, 9.51326, 25.143, 50.5208],
    10: [0.0066391, 0.0383377, 0.294219, 1.00794, 2.50992, 8.96365, 22.4797, 49.2331],
    12: [0.0072405, 0.0349922, 0.271527, 1.04127, 2.48292, 8.81052, 21.6285, 47.8551],
}

# =========================
# 1. График: время от размера матрицы
# =========================
plt.figure(figsize=(10, 6))

for t in threads:
    plt.plot(matrix_sizes, times[t], marker='o', label=f'{t} поток(ов)')

plt.xlabel('Размер матрицы N')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения от размера матрицы')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 2. График: время от количества потоков
# =========================
plt.figure(figsize=(10, 6))

for i, n in enumerate(matrix_sizes):
    y = [times[t][i] for t in threads]
    plt.plot(threads, y, marker='o', label=f'N = {n}')

plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения от количества потоков')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 3. Тепловая карта
# =========================
data = np.array([times[t] for t in threads])

plt.figure(figsize=(10, 6))
im = plt.imshow(data, aspect='auto', origin='lower')

plt.colorbar(im, label='Время выполнения (сек)')
plt.xticks(range(len(matrix_sizes)), matrix_sizes)
plt.yticks(range(len(threads)), threads)
plt.xlabel('Размер матрицы N')
plt.ylabel('Количество потоков')
plt.title('Тепловая карта времени выполнения')
plt.tight_layout()
plt.show()