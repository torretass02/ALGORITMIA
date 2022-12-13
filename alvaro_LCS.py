import numpy as np
from typing import List, Tuple, Dict, Callable, Iterable, Union

def LCS (x:str, y:str)->np.ndarray:
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

    return b

def LCS_print (B:np.array, x:str, i:int, j:int, string:str)->str:
    if i==0 or j==0:
        string = string[::-1]
        print("1", string)
        return string
    if B[i, j] == 'D':
        string = string + x[i-1]
        print("2", string)
        u = LCS_print (B, x, i-1, j-1, string)
    elif B[i, j] == 'U':
        print("3", string)
        LCS_print (B, x, i-1, j, string)
    elif B[i, j] == 'L':
        print("4", string)
        LCS_print (B, x, i, j-1, string)

y = "BDCABA"
x = "ABCBDAB"

array = LCS(x, y)
vacio = ""

string_final = LCS_print(array, x, len(x), len(y), vacio)
