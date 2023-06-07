import random
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt

def run_q_ising(q: int, nodes: np.ndarray, t: float):
	m = np.empty(100)

	N = nodes.size
	
	# Mc = 1000 (duze kroki Monte Carlo)
	for i in range(1000):
		n = 0

		for node in range(N):
			lobby = np.random.randint(N, size = q)

			energy = -1 * nodes[node] * np.sum(nodes[lobby])

			prob = min(1, np.exp(2 * energy / t))

			if np.random.rand() < prob:
				nodes[node] *= -1

			if nodes[node] == 1:
				n += 1

		if i > 900: # dla ostatnich 100 duzych krokow
			m[i - 901] = 2 * n / N - 1

	return abs(np.average(m))

def sim_complete(q: int, n: int, t: (float, float)):
	T = np.linspace(t[0], t[1], num = 50)
	M = np.empty((2, 50))

	for i, temp in np.ndenumerate(T):
		print(str(i[0]) + " complete")

		g_uniform = np.ones(n)
		g_random = np.random.choice([-1, 1], size = n)

		M[0][i[0]] = run_q_ising(q, g_uniform, temp)
		M[1][i[0]] = run_q_ising(q, g_random, temp)

	data = np.column_stack((T, M[0], M[1]))
	data = np.asarray(data)

	np.savetxt(str(q) + 'q_ising_complete.csv', data, delimiter = ',')

def main():
    p1 = Process(target = sim_complete, args = (3, 500, [1, 2]))
    p2 = Process(target = sim_complete, args = (4, 500, [1.3, 2.3]))
    p3 = Process(target = sim_complete, args = (5, 500, [2.7, 3.7]))
    p4 = Process(target = sim_complete, args = (6, 500, [3.3, 4.3]))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

if __name__ == "__main__":
    main()
