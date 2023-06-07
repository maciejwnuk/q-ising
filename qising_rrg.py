import random
from multiprocessing import Process
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

def run_q_ising(q: int, g: ig.Graph, t: float):
	N = g.vcount()
	
	# Mc = 1000 (duze kroki Monte Carlo)
	for i in range(1000):
		for v in g.vs:
			lobby = np.random.choice(v.neighbors(), q, False)

			energy = 0

			for nbr in lobby:
				energy += nbr["spin"]

			energy *= -1 * v["spin"]

			prob = min(1, np.exp(2 * energy / t))

			if np.random.rand() < prob:
				v["spin"] *= -1

	n = 0

	for v in g.vs:
		if v["spin"] == 1:
			n += 1

	return abs(2 * n / N - 1)

def sim_rrg(q: int, n: int, t: (float, float), k: int):
	T = np.linspace(t[0], t[1], num = 50)
	M = np.empty((2, 50))

	for i, temp in np.ndenumerate(T):
		g_uniform = ig.Graph.K_Regular(n, k)
		g_random = ig.Graph.K_Regular(n, k)

		g_uniform.vs["spin"] = np.ones(n)
		g_random.vs["spin"] = np.random.choice([-1, 1], size = n)

		M[0][i[0]] = run_q_ising(q, g_uniform, temp)
		M[1][i[0]] = run_q_ising(q, g_random, temp)

	data = np.column_stack((T, M[0], M[1]))
	data = np.asarray(data)

	np.savetxt(str(q) + 'q_ising_rrg_k' + str(k) + '.csv', data, delimiter = ',')

def main():
	processes = [Process(target = sim_rrg, args = (3, 1000, [1, 2], 10)), Process(target = sim_rrg, args = (3, 1000, [1, 2], 50)), 
				 Process(target = sim_rrg, args = (4, 1000, [1.5, 2.6], 10)), Process(target = sim_rrg, args = (4, 1000, [1.5, 2.6], 50)), 
				 Process(target = sim_rrg, args = (5, 1000, [2.7, 3.7], 10)), Process(target = sim_rrg, args = (5, 1000, [2.7, 3.7], 50)), 
				 Process(target = sim_rrg, args = (6, 1000, [3.6, 4.6], 10)), Process(target = sim_rrg, args = (6, 1000, [3.6, 4.6], 50))]

	for process in processes:
		process.start()

	for process in processes:
		process.join()

if __name__ == "__main__":
    main()
