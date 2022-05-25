import numpy as np
import time
import random
import sys
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Assignment4A:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        def inverse(mat):
            issquare = lambda m : len(m[0]) == len(m)

            if not issquare(mat):
                return ZeroDivisionError
            number_of_rows = len(mat)
            inverse_mat = [[0 for i in range(number_of_rows)] for i in range(number_of_rows)]
            for i in range(0, number_of_rows): inverse_mat[i][i] = 1

            for i in range(number_of_rows):
                if mat[i][i] == 0:
                    return ZeroDivisionError

                if mat[i][i] != 1:
                    factor = mat[i][i]
                    divide(mat, i, factor)
                    divide(inverse_mat, i, factor)

                for j in range(number_of_rows):
                    if i != j:
                        if mat[j][i] != 0:
                            factor = mat[j][i]
                            mat[j] = [np.float64(k - factor * l) for k, l in zip(mat[j], mat[i])]
                            inverse_mat[j] = [np.float64(k - factor * l) for k, l in zip(inverse_mat[j], inverse_mat[i])]

            return inverse_mat

        def divide(mat, row, factor):
            for i in range(len(mat[row])):
                mat[row][i] /= np.float64(factor)

        def get_avg_y(f,x):
            l = 0
            for i in range(1000):
                l+= np.float64(f(x))
            return np.float64(l/1000)

        tmp = maxtime * 40
        x_points = np.linspace(a,b,tmp)
        b = []
        A = []

        for x in x_points:
            temp = []
            for i in range(d,-1,-1):
                temp.append(x**i)
            A.append(temp)
            b.append([np.float64(get_avg_y(f,x))])

        b = torch.tensor(b)
        A = torch.tensor(A)

        def sol(A, b):
            transposed = torch.transpose(A,0,1)
            At_A = torch.matmul(transposed, A)
            At_b = torch.matmul(transposed, b)
            return torch.matmul(torch.tensor(inverse(At_A.tolist())),torch.tensor(At_b))

        return np.poly1d(sol(A,b).flatten())


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    # def test_return(self):
    #     f = NOISY(0.01)(poly(1,1))
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     print("Time:",T)
    #     self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
    #
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1,1,1,1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print("MSE:",mse)


if __name__ == "__main__":
    unittest.main()
