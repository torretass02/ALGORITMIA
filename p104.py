'Hugo Torres Martínez y Luis Rodríguez Moreno'

from heapq import heapify
from attr import has
import numpy as np
from typing import List, Callable


'MULTIPLICACIÓN DE MATRICES'


def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray:
    """
        Realiza la multiplicación de dos matrices pasadas por argumento.

        Args:
            m_1: Primera matriz por multiplicar
            m_2: Segunda matriz por multiplicar

        Return:
            Matriz resultado de la multiplicación de las anteriores matrices.


    """
    n_rows, n_interm, n_columns = \
        m_1.shape[0], m_2.shape[0], m_2.shape[1]
    m_product = np.zeros((n_rows, n_columns))

    for p in range(n_rows):
        for q in range(n_columns):
            for r in range(n_interm):
                m_product[p, q] += m_1[p, r] * m_2[r, q]
    return m_product


'BUSQUEDA BINARIA CON RECURSION'


def rec_bb(t: list, f: int, l: int, key: int) -> int:
    """
        Busca la posición de un elemento en un array, siguiendo el procedimiento de la búsqueda binaria con recursión.

        Args:
            t: Lista en la que se va a realizar la búsqueda binaria.
            f: Primer índice.
            l: Último índice.
            key: Entero a buscar en el array.

        Return:
            Si encuentra la "key" en el array, devuelve su posición. Si no la encuentra, devuelve None.


    """
    if f > l:
        return
    if f == l:
        if key == t[f]:
            return f
        else:
            return None

    mid = (f + l) // 2

    if key == t[mid]:
        return mid
    elif key < t[mid]:
        return rec_bb(t, f, mid - 1, key)
    else:
        return rec_bb(t, mid + 1, l, key)


'BUSQUEDA BINARIA SIN RECURSION'


def bb(t: List, f: int, l: int, key: int) -> int:
    """
        Busca la posición de un elemento en un array, siguiendo el procedimiento de la búsqueda binaria sin recursión.

        Args:
            t: Lista en la que se va a realizar la búsqueda binaria.
            f: Primer índice.
            l: Último índice.
            key: Entero a buscar en el array.

        Return:
            Si encuentra la "key" en el array, devuelve su posición.


    """

    while f <= l:
        mid = (f + l) // 2

        if key == t[mid]:
            return mid
        elif key < t[mid]:
            l = mid - 1
        else:
            f = mid + 1


'MIN HEAP'


def min_heapify(h: np.ndarray, i: int):
    """
        Aplica la operación de heapify al elemento situado en la posición "i" en el array "h".

        Args:
            h: Array en la que se ejecuta la operación.
            i: Posición del array en la que se ejecuta el heapify.


    """
    while 2 * i + 1 < len(h):
        n_i = i

        if h[i] > h[2 * i + 1]:
            n_i = 2 * i + 1
        if 2 * i + 2 < len(h) and h[i] > h[2 * i +
                                           2] and h[2 * i + 2] < h[n_i]:
            n_i = 2 * i + 2

        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i
        else:
            return


'INSERT MIN HEAP'


def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    """
        Inserta el entero "k" en el min heap contenido en "h".

        Args:
            h: Array en el q se va a ejecutar la operación de inserción.
            k: Entero a introducir en el min heap.

        Return:
            Nuevo min heap con el entero "k" introducido en él.


    """

    if isinstance(h, type(None)):
        h = [k]
        return h

    h = np.append(h, k)
    j = np.size(h) - 1

    while j >= 1 and h[(j - 1) // 2] > h[j]:
        h[(j - 1) // 2], h[j] = h[j], h[(j - 1) // 2]
        j = (j - 1) // 2

    return h


'CREATE MIN HEAP'


def create_min_heap(h: np.ndarray):
    """
        Crea un min heap sobre el array de Numpy pasado como argumento.

        Args:
            h: Array en la que se realiza la operación min heap.


    """
    j = ((len(h) - 1) - 1) // 2
    while j > -1:
        min_heapify(h, j)
        j -= 1
    return h


'PQ INI'


def pq_ini():
    """
        Inicializa una cola de prioridad vacía.

    """
    h = []
    return h


'PQ INSERT'


def pq_insert(h: np.ndarray, k: int) -> np.ndarray:
    """
        Inserta el elemento "k" en la cola de prioridad "h".

        Args:
            h: Array en el q se va a ejecutar la operación de inserción.
            k: Elemento a insertar en la cola de prioridad.

        Return:
            Nueva cola de prioridad con el elemento nuevo insertado.


    """
    h = insert_min_heap(h, k)
    return h


'PQ REMOVE'


def pq_remove(h: np.ndarray) -> tuple([int, np.ndarray]):
    """
        Elimina el elemento con el menor valor de prioridad de "h".

        Args:
            h: Array en el q se va a ejecutar la operación de extracción.

        Return:
            El elemento eliminado y la nueva cola de prioridad.


    """
    raiz = h[0]
    h[0] = h[len(h) - 1]
    h = h[:-1]
    min_heapify(h, 0)

    return (int(raiz), h)


'SELECT MIN HEAP'


def select_min_heap(h: np.ndarray, k: int) -> int:
    """
        Realiza una operación que consiste en encontrar el valor que ocuparía la posición "k" si "h" se tratase de un array ordenado.

        Args:
            h: Array en el q se va a ejecutar la operación.
            k: Posición del array que debe ocupar el valor devuelto por la función.

        Return:
            Elemento que ocuparía la posición "k" si "h" fuese un array ordenado.


    """
    h = create_min_heap(h)
    for i in range(k - 1):
        e, h = pq_remove(h)

    return h[0]
