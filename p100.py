'NUESTRA FORMA'
from heapq import heapify
from attr import has
import numpy as np

def matrix_multiplication(m1: np.ndarray, m2 : np.ndarray) :
    if len (m1[0]) == len (m2): 
        m3 = []
        for i in range (len (m1)):
            m3.append ([])
            for j in range (len (m2[0])):
                m3 [i].append(0)
        
        for i in range (len (m1)):
            for j in range (len (m2[0])):
                for k in range(len(m1[0])):
                    m3[i][j] += m1[i][j] * m2[k][j]
        return m3
    else: 
        return None


'BUSQUEDA BINARIA CON RECURSION'

def rec_bb(t: list, f: int, l: int, key: int):
    if f>l:
        return
    if f == l:
        if key == t[f]:
            return f
        else:
            return None
    
    mid = (f + l)//2
    
    if key == t[mid]:
        return mid
    elif key < t[mid]:
        return rec_bb(t, f, mid-1, key)
    else:
        return rec_bb(t, mid+1, l, key)

'BUSQUEDA BINARIA SIN RECURSION'

from traitlets import List
def bb(t: List, f: int, l: int, key: int):

    while f <= l:
        mid = (f + l)//2
        
        if key == t[mid]:
            return mid
        elif key < t[mid]:
            l = mid-1
        else:
            f = mid+1

'MIN HEAP'

def min_heapify(h: np.ndarray, i: int):
    while 2*i+1 < len(h):
        n_i = i
        
        if h[i] > h[2*i+1]:
            n_i = 2*i+1
        if 2*i+2 < len(h) and h[i] > h[2*i+2] and h[2*i+2] < h[n_i]:
            n_i = 2*i+2

        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i
        else:
            return

'INSERT MIN HEAP'

def insert_min_heap(h: np.ndarray, k:int)-> np.ndarray:

    if isinstance(h,type(None)):
        h=[k]
        return h

    h = np.append(h,k)
    j = np.size(h) - 1
    
    while j >= 1 and h[(j-1) // 2] > h[j]:
        h[(j-1) // 2], h[j] = h[j], h[(j-1) // 2]
        j = (j-1) // 2  
    
    return h
        
'CREATE MIN HEAP'

def create_min_heap (h:np.ndarray):
    
    j =((len(h)-1) -1)//2
    while j >-1:
        min_heapify (h, j)
        j -= 1
    return h

'PQ INI'

def pq_ini():
    h = []
    return h

'PQ INSERT'

def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    h = insert_min_heap(h,k)
    return h

'PQ REMOVE'

def pq_remove(h: np.ndarray)-> tuple([int, np.ndarray]):
    
    raiz = h[0]
    h[0] =  h[len(h)-1]
    h =h[:-1]
    min_heapify(h, 0)

    return (int(raiz), h)

'SELECT MIN HEAP'

def select_min_heap(h: np.ndarray, k: int)-> int:
