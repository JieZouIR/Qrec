import numpy as np
import torch
from torch.nn import functional

def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


def mmread(R, type='float32'):
    row = R.row.astype(int)
    col = R.col.astype(int)
    val = torch.from_numpy(R.data.astype(type))
    index = torch.from_numpy(np.row_stack((row, col)))
    m, n = R.shape
    return torch.sparse.FloatTensor(index, val, torch.Size([m, n]))

def csr2test(test):
    return {str(r): {str(test.indices[ind]): int(1)
                     for ind in range(test.indptr[r], test.indptr[r + 1])}
            for r in range(test.shape[0]) if test.indptr[r] != test.indptr[r + 1]}

def sort2query(run):
    m, n = run.shape
    return {str(i): {str(int(run[i, j])): float(1.0 / (j + 1)) for j in range(n)} for i in range(m)}

def sort2query_vector(run):
    return {str(int(run[j])): float(1.0 / (j + 1)) for j in range(len(run))}

