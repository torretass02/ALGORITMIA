import numpy as np
from itertools import permutations
from typing import List, Dict, Callable, Iterable


def init_cd(n: int) -> np.ndarray:
    arr = np.ones(n).astype(int)*-1
    return arr


def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    x = find(rep_1, p_cd)
    y = find(rep_2, p_cd)

    if x == y:
        return -1

    if p_cd[y] < p_cd[x]:
        p_cd[x] = y
        ret = y
    elif p_cd[y] > p_cd[x]:
        p_cd[y] = x
        ret = x
    else:
        p_cd[y] = x
        p_cd[x] -= 1
        ret = x
    return ret


def find(ind: int, p_cd: np.ndarray) -> int:
    z = ind

    while p_cd[z] >= 0:
        z = p_cd[z]

    while p_cd[ind] >= 0:
        y = p_cd[ind]
        p_cd[ind] = z
        ind = y

    return z


def cd_2_dict(p_cd: np.ndarray) -> dict:
    d = {}

    for i in range(len(p_cd)):
        if p_cd[i] < 0:
            d[i] = []

    for i in range(len(p_cd)):
        f = find(i, p_cd)
        d[f].append(i)

    return d


def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    h = np.random.randint(w_max, size=(n_nodes, n_nodes))
    np.fill_diagonal(h, 0)

    return h


def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    num_cities = dist_m.shape[0]
    circuit = [node_ini]

    while len(circuit) < num_cities:
        current_city = circuit[-1]
        options = list(np.argsort(dist_m[current_city]))

        for city in options:
            if city not in circuit:
                circuit.append(city)
                break

    return circuit + [node_ini]


def len_circuit(circuit: list, dist_m: np.ndarray) -> int:
    longitud = 0

    for i in range(len(dist_m)):
        longitud += dist_m[circuit[i]][circuit[i+1]]

    return longitud


def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    circuitos = []
    longitudes = []

    for i in range(len(dist_m)):
        circuitos.append(greedy_tsp(dist_m, i))
        longitudes.append(len_circuit(circuitos[i], dist_m))

    min = np.amin(longitudes)
    index = longitudes.index(min)

    return circuitos[index]

def exhaustive_tsp(dist_m: np.ndarray)-> list:
  c = []
  l = []
  h = []

  for i in range(len(dist_m)):
    c.append(i)

  p = list(permutations(c))

  for i in p:
    l.append(list(i))

  for i in range(len(l)):
    l[i].append(l[i][0])

  min = len_circuit(l[0], dist_m)
  h = list(np.copy(l[0]))
  
  for i in l[1:]:
    aux = len_circuit(i, dist_m)
    if aux < min:
      min = aux
      h = list(np.copy(i))

  return h