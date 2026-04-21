import matplotlib.pyplot as plt

matrix_sizes = [100, 200, 400, 800, 1200, 1600, 2000]
execution_times = [0.0452226, 0.328577, 2.4552, 15.9219, 48.3915, 119.406, 243.757]

plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, execution_times, marker='o')
plt.xlabel("Размер матрицы N")
plt.ylabel("Время выполнения (сек)")
plt.title("Зависимость времени выполнения от размера матрицы")
plt.grid(True)
plt.show()