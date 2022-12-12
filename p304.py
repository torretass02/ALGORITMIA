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
    n_group = len(t) // 5
    index_median = 5 // 2 
    cutoff = 5
    sublists =  [t[j:j+ 5] for j in range(0, len(t), 5)][:n_group]
    medians = [sorted(sub)[index_median] for sub in sublists]

    if len(medians) <= cutoff:
        pivot = sorted(medians)[len(medians)//2]
    else:
        pivot = QSelect5 (medians, len(medians)//2)
        
    return pivot

def QSelect5 (data:List, k:int) -> Union[int, None]:
    cutoff = 5
    
    if len(data) <= cutoff:
        return sorted(data)[k-1]
    
    pivot_value = pivot5(data)
    
    s_1, pivot_value, s_2 = split_pivot(data, pivot_value)

    pivot_index = len(s_1)

    if (k-1) == pivot_index  :
        return pivot_value
    elif (k-1) < pivot_index :
        return QSelect5 (s_1, k)
    else:
        return QSelect5 (s_2, k-pivot_index-1)

def qsel5_nr (data:List, k:int) -> Union[int, None]:
    cutoff = 5

    if len(data) == 1 and k == 0:
        return data[0]
    
    if k<0 or k>len(data)-1:
        return None

    if len(data) <= cutoff:
        return sorted(data)[k-1]
    else:
        pivot_value = pivot5(data)
    
        s_1, pivot_value, s_2 = split_pivot(data, pivot_value)

        pivot_index = len(s_1)

        if (k-1) == pivot_index  :
            return pivot_value
        elif (k-1) < pivot_index :
            return qsel5_nr (s_1, k)
        else:
            return qsel5_nr (s_2, k-pivot_index-1)
