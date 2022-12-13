import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union

def split(t: List) -> Tuple[List, int, List]:
    mid = t[0]
    t_l = [u for u in t if u < mid]
    t_r = [u for u in t if u > mid]
    return (t_l, mid, t_r)

def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    if len(t) == 1 and k == 0:
        return t[0]
    if k<0 or k>len(t)-1:
        return None
        
    t_l, mid, t_r = split(t)
    
    m = len(t_l)
    if k == m:
        return mid
    elif k < m:
        return qsel(t_l, k)
    else:
        return qsel(t_r, k-m-1)
    
def qsel_nr(t: np.ndarray, k: int)-> Union[int, None]:
    if(k < 0 or t is None):
        return None

    if k<0 or k>len(t)-1:
        return None

    if(len(t)==1):
        return t[0]
    else:
        t1, p, t2 = split(t)
        m = len(t1)

        if k==m:
            ret = p
        elif k<m:
            ret = qsel_nr(t1, k)
        elif k>m:       
            ret = qsel_nr(t2, k-m-1)

    return ret

def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    t_l = [u for u in t if u < mid]
    t_r = [u for u in t if u > mid]
    return (t_l, mid, t_r)

def pivot5(t: np.ndarray)-> int:
    if len(t)>0 and len(t)<=5:
        return int(np.median(t))
    elif len(t)==0:
        return None
    else:
        return int(np.median([pivot5(t[i:i+5]) for i in range(0, len(t), 5)]))

def qsel5_nr (t:np.ndarray, k:int) -> Union[int, None]:
    
    if len(t)==0  or k>len(t)-1 or k<0:
        return None
    while True:
        pivot_v = pivot5(t)

        if pivot_v == None:
            return None
        
        s_1, piv, s_2 = split_pivot(t, pivot_v)

        if k == len(s_1) :
            return piv
        elif k < len(s_1) :
           t = s_1
        else:
            t = s_2
            k = k - len(s_1) - 1

def edit_distance(str_1: str, str_2: str)-> int:
    D = np.zeros((len(str_1)+1, len(str_2)+1), dtype=int)
    D[0, 1:] = range(1, len(str_2)+1)
    D[1:, 0] = range(1, len(str_1)+1)
    
    for i in range(1, len(str_1)+1):
        for j in range(1, len(str_2)+1):
            if (str_1[i-1] == str_2[j-1]):
                D[i,j] = D[i-1, j-1]
            else :
                D[i, j] = min(D[i-1, j-1]+1, D[i-1, j]+1, D[i, j-1]+1)
    
    return D[-1, -1]

def max_subsequence_length(str_1: str, str_2: str)-> int:
    e = np.zeros((len(str_1)+1, len(str_2)+1), dtype=int)
    
    for i in range(1, len(str_1)+1):
        for j in range(1, len(str_2)+1):
            if (str_1[i-1] == str_2[j-1]):
                e[i,j] = 1 + e[i-1, j-1]    
            else :
                e[i, j] = max(e[i-1, j], e[i, j-1])
    return e[-1,-1]

def qsort_5(t:np.ndarray) -> np.ndarray:  
    if len(t)<= 5:
        return np.sort(t)
    else:
        m = pivot5(t)
        l, piv, r = split_pivot(t, m)
        return np.concatenate((qsort_5(l), np.array([piv]), qsort_5(r)))
    
def max_common_subsequence (x:str, y:str)->str:
    e = np.zeros((len(x)+1, len(y)+1), dtype=int)
    b = np.empty((len(x)+1, len(y)+1), dtype=str)
    
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            if (x[i-1] == y[j-1]):
                e[i,j] = 1 + e[i-1, j-1]
                b[i,j] = 'D'
                  
            else :
                inputlist = (e[i-1, j], e[i, j-1])
                max_value = max(inputlist)
                max_index=inputlist.index(max_value)
                e[i, j], index = max_value, max_index
                b[i, j] = 'L' if index else 'U'


    def LCS_print (B:np.array, x:str, i:int, j:int, string:str)->str:
        if i==0 or j==0:
            string = string[::-1]
            return string
        if B[i, j] == 'D':
            string = string + x[i-1]
            return LCS_print (B, x, i-1, j-1, string)
        elif B[i, j] == 'U':
            return LCS_print (B, x, i-1, j, string)
        elif B[i, j] == 'L':
            return LCS_print (B, x, i, j-1, string)

    vacio = "" 
    return LCS_print(b, x, len(x), len(y), vacio)


    