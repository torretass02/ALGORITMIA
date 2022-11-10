'Hugo Torres Martínez y Luis Rodríguez Moreno'

import numpy as np
from itertools import permutations
from typing import List, Dict, Callable, Iterable

'INIT CD'


def init_cd(n: int) -> np.ndarray:
    """
        Función que devuelve un array con valores -1 en las posiciones {0, 1, ..., n-1}.
        Args:
            n: Tamaño del array.
        Return:
            Array con valores -1 en las posiciones {0, 1, ..., n-1}.
    """

    arr = np.ones(n).astype(int)*-1
    return arr


'UNION'


def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    """
        Realiza la operacion "Union" de dos conjuntos disjuntos.
        Args:
            rep_1: Representante del primer conjunto disjunto.
            rep_2: Representante del segundo conjunto disjunto.
            p_cd: Array en el cuál se almacena el conjunto disjunto resultante. 
        Return:
            Representante del conjunto obtenido como la union por rangos de los representados por los índices rep_1, rep_2 en el CD almacenado en el array p_cd.
    """

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


'FIND'


def find(ind: int, p_cd: np.ndarray) -> int:
    """
        Realiza la operacion "Find" de un conjunto disjunto.
        Args:
            ind: Índice del cual se quiere obtener el representante del CD.
            p_cd: CD del cual se quiere obtener el representante.
        Return:
            Devuelve el representante del índice ind en el CD almacenado en p_cd realizando compresion de caminos.
    """

    z = ind

    while p_cd[z] >= 0:
        z = p_cd[z]

    while p_cd[ind] >= 0:
        y = p_cd[ind]
        p_cd[ind] = z
        ind = y

    return z


'DICCIONARIO CD'


def cd_2_dict(p_cd: np.ndarray) -> dict:
    """
        Función que se encarga de identificar visualmente los subconjuntos de un CD.
        Args:
            p_cd: CD que se quiere representar en el diccionario.
        Return:
            Un diccionario cuyas claves sean los representantes de los subconjuntos del CD y donde el valor de la clave u del dict sea una lista con los miembros del subconjunto representado por u , incluyendo, por supuesto el propio u .
    """

    d = {}

    for i in range(len(p_cd)):
        if p_cd[i] < 0:
            d[i] = []

    for i in range(len(p_cd)):
        f = find(i, p_cd)
        d[f].append(i)

    return d


'COMPONENTES CONEXAS'


def ccs(n: int, l: list) -> dict:
    """
        Función que se encarga de identificar las componentes conexas de un grafo dado por argumento.
        Args:
            n: 
            list: Grafo del que se van a extraer las componentes conexas. 
        Return:
            Las componentes conexas de un tal grafo.
    """

    d_cc = {}

    h = init_cd(n)

    for u, v in l:
        r_u = find(u, h)
        r_v = find(v, h)

        if r_u != r_v:
            union(r_u, r_v, h)

    d_cc = cd_2_dict(h)
    return d_cc


'MATRIZ DE DISTANCIAS'


def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    """
        Función que genera una matriz de distancias de n_nodes nodos con distancias máximas w_max.
        Args:
            n_nodes: Número de nodos de la matriz.
            w_max: Valor máximo que puede tener la distancia de cada nodo de la matriz.
        Return:
            Matriz de distancias.
    """

    h = np.random.randint(w_max, size=(n_nodes, n_nodes))
    np.fill_diagonal(h, 0)

    return h


'GREEDY TSP'


def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    """
        Función que a partir de un nodo inicial calcula el circuito que pase por el resto de nodos de menor distancia.
        Args:
            dist_m: Matriz de distancias de la que se quiere obtener el circuito.
            node_ini: Nodo en el que se empezará y finalizará el circuito.
        Return:
            Circuito que siga las pautas del "Vecino más cercano".
    """

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


'LONGITUD DE CIRCUITO'


def len_circuit(circuit: list, dist_m: np.ndarray) -> int:
    """
        Función que obtiene la longitud total de un circuito dado por argumento.
        Args:
            circuit: Circuito del que se quiere obtener su longitud total.
            dist_m: Matriz de distancias a la que pertenece el circuito circuit.
        Return:
            Longitud del circuito dado por argumento. 
    """

    longitud = 0

    for i in range(len(dist_m)):
        longitud += dist_m[circuit[i]][circuit[i+1]]

    return longitud


'REPEATED GREEDY TSP'


def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    """
        Función que, mediante el uso de greedy, obtiene todos los posibles circuitos y devuelve el que menos longitud tenga.
        Args:
            dist_m: Matriz de distancias de la que se quieren obtener los circuitos.
        Return:
            Longitud del circuito dado por argumento. 
    """

    circuitos = []
    longitudes = []

    for i in range(len(dist_m)):
        circuitos.append(greedy_tsp(dist_m, i))
        longitudes.append(len_circuit(circuitos[i], dist_m))

    min = np.amin(longitudes)
    index = longitudes.index(min)

    return circuitos[index]


'EXHAUSTIVE TSP'


def exhaustive_tsp(dist_m: np.ndarray) -> list:
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
