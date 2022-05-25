import numpy as np
import time
import random
from assignment2 import Assignment2

class Assignment3():
    def __init__(self):
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:

        def trapz(g, a, b, N):
            N=N-1
            x_list = np.linspace(a, b, N + 1)
            y = np.array([g(xx) for xx in x_list])
            y_right = y[1:]
            y_left = y[:-1]
            deriv = (b - a) / N
            area = (deriv / 2) * np.sum(y_right + y_left)
            return np.float32(area)


        def simps(g, a, b, N):
            N=N-1
            if N % 2 == 1:
                N = N-1
            deriv = (b - a) / N
            x = np.linspace(a, b, N + 1)
            y = np.array([g(xx) for xx in x])
            area = deriv / 3. * np.sum(y[0:-1:2]+ 4. * y[1::2] + y[2::2])
            return np.float32(area)

        if n < 3: return trapz(f,a,b,n)
        else: return simps(f,a,b,n)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        ass2 = Assignment2()
        result = 0.0
        g = lambda x: f1(x)-f2(x)
        all_points = ass2.intersections(f1,f2,1,100)
        if len(all_points) < 2: return None
        else:
            for i in range(len(all_points)-1):
                temp = 1
                if all_points[i+1] - all_points[i] > 1 : int(all_points[i+1] - all_points[i])+1
                result += np.abs(self.integrate(g,all_points[i],all_points[i+1],50*temp))
        return np.float32(result)



##########################################################################
#
#
# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
# from numpy import sin,cos
#
# class TestAssignment3(unittest.TestCase):
#
#     # def test_integrate_float32(self):
#     #     ass3 = Assignment3()
#     #     # f1 = np.poly1d([-1, 0, 1])
#     #     f1 = lambda x : np.sin(x)
#     #     r = ass3.integrate(f1, 1, 5, 100)
#     #     self.assertEqual(r.dtype, np.float32)
#     #
#     # def test_integrate_hard_case(self):
#     #     ass3 = Assignment3()
#     #     f1 = strong_oscilations()
#     #     r = ass3.integrate(f1,  0.09, 10, 20)
#     #     print(r)
#     #     true_result = -7.78662 * 10 ** 33
#     #     print(true_result)
#     #     self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))
#
#     def test_area(self):
#         T = time.time()
#         f1 = lambda x: cos(x)
#         f2 = lambda x: pow(x, 2) - 3 * x + 2
#         f3 = lambda x: sin(x)
#         ass3 = Assignment3()
#         r=ass3.areabetween(f3,f1)
#         true_result = 1.0258
#         print(time.time() - T)
#         print("error:", abs((r - true_result) / true_result))
#         self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))
#
# if __name__ == "__main__":
#     unittest.main()
