import numpy as np
import time
import random
from collections.abc import Iterable

def secant(f,a,b,err,N=15):
    if np.sign(f(a)) == np.sign(f(b)):
        return
    if np.abs(f(a)) <= err: return a
    if np.abs(f(b)) <= err: return b
    for n in range(1,N+1):
        s = a - f(a)*(b - a)/(f(b) - f(a))
        f_s = f(s)
        if f(a)*f_s < 0:
            b = s
        elif f(b)*f_s < 0:
            a = s
        elif f_s <= err:
            return s
        else:
            return
    return a - f(a)*(b - a)/(f(b) - f(a))

class Assignment2:
    def __init__(self):
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        sections = np.linspace(a,b,int((b-a)/(maxerr*2)))
        result = []
        g= lambda x: f1(x)-f2(x)
        for i in range(len(sections)-1):
            temp = secant(g,sections[i],sections[i+1],maxerr)
            if temp != None: result.append(temp)
        return result


##########################################################################

#
# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
# #
# #
# class TestAssignment2(unittest.TestCase):
#
#     def test_sqr(self):
#         ass2 = Assignment2()
#
#         f1 = np.poly1d([1,-1, 1, 0])
#         f2 = np.poly1d([1, 0, 0])
#
#         X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#         for x in X:
#             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
#     def test_poly(self):
#
#         ass2 = Assignment2()
#
#         f1, f2 = randomIntersectingPolynomials(100)
#
#         X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#         print(X)
#         for x in X:
#             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
# #
#
# if __name__ == "__main__":
#     unittest.main()
