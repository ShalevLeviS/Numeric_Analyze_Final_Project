import numpy as np
import time
import random
import copy
class Assignment1:
    def __init__(self):
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:

        def newton_rhapson(f, df, x0, tol):
            i = 0
            while abs(f(x0)) > tol or i < 10:
                x0 = x0 - f(x0) / df(x0)
                i+=1
            return x0

        def Thomas_equation_solver(a, b, c, d):
            length = len(d)
            ac, bc, cc, dc = map(np.array, (a, b, c, d))
            for it in range(1, length):
                mc = ac[it - 1] / bc[it - 1]
                bc[it] = bc[it] - mc * cc[it - 1]
                dc[it] = dc[it] - mc * dc[it - 1]

            xc = bc
            xc[-1] = dc[-1] / bc[-1]

            for il in range(length - 2, -1, -1):
                xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

            return xc

        def get_bezier_coefficents(points):
            n = len(points) - 1

            Coef_matrix = 4 * np.identity(n)
            np.fill_diagonal(Coef_matrix[1:], 1)
            np.fill_diagonal(Coef_matrix[:, 1:], 1)
            Coef_matrix[0, 0] = 2
            Coef_matrix[n - 1, n - 1] = 7
            Coef_matrix[n - 1, n - 2] = 2
            res = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
            res[0] = points[0] + 2 * points[1]
            res[n - 1] = 8 * points[n - 1] + points[n]

            def find_abc(C):
                a = []
                b = []
                c = []
                for i in range(n-1):
                    c.append(C[i][i + 1])
                    b.append(C[i][i])
                    a.append(C[i + 1][i])
                b.append(C[n-1][n-1])
                return a, b, c

            a, b, c = find_abc(Coef_matrix)

            first_mat = Thomas_equation_solver(a,b,c, res)
            sec_mat = [0] * n
            for i in range(n - 1):
                sec_mat[i] = 2 * points[i + 1] - first_mat[i + 1]
            sec_mat[n - 1] = (first_mat[n - 1] + points[n]) / 2

            return first_mat, sec_mat

        def get_bezier_cubic_points(points):
            first_mat, sec_mat = get_bezier_coefficents(points)
            return [[points[i], first_mat[i], sec_mat[i], points[i + 1]] for i in range(len(points) - 1)]

        if (n == 1): return lambda x: f((a+b)/2)
        else:
            x_points = np.linspace(a,b,n)
            y_points = np.array([f(x) for x in x_points])
            all_curves_x = get_bezier_cubic_points(x_points)
            all_curves_y = get_bezier_cubic_points(y_points)
            all_curves = []
            for i in range(len(all_curves_x)):
                p0 = [all_curves_x[i][0],all_curves_y[i][0]]
                p1 = [all_curves_x[i][1], all_curves_y[i][1]]
                p2 = [all_curves_x[i][2], all_curves_y[i][2]]
                p3 = [all_curves_x[i][3], all_curves_y[i][3]]
                all_curves.append([p0,p1,p2,p3])

        def GetYfromX(P0, P1, P2, P3, x):
            coefficients = [-P0[0] + 3 * P1[0] - 3 * P2[0] + P3[0], 3 * P0[0] - 6 * P1[0] + 3 * P2[0],
                            -3 * P0[0] + 3 * P1[0], P0[0] - x]
            x0 = 1
            func = lambda x: coefficients[0] * x ** 3 + coefficients[1] * x ** 2 + coefficients[2] * x + coefficients[3]
            deriv = lambda x: 3 * coefficients[0] * x ** 2 + 2 * coefficients[1] * x + coefficients[2]
            i = 0
            root = newton_rhapson(func, deriv, x0, 0.001)
            while root > 1 or root < 0:
                if i == 5: break
                root = newton_rhapson(func, deriv, x0, 0.001)
                x0 = x0 / 2
                i += 1

            return (1 - root) ** 3 * P0[1] + 3 * (1 - root) ** 2 * root * P1[1] + 3 * (1 - root) * root ** 2 * P2[
                1] + root ** 3 * P3[1]


        def g(x):
            first = 0
            last = len(x_points)-1
            while first <= last:
                midpoint = (first + last) // 2

                if x_points[midpoint] <= x and x <= x_points[midpoint + 1]:
                    return GetYfromX(all_curves[midpoint][0],all_curves[midpoint][1],all_curves[midpoint][2],all_curves[midpoint][3],x)
                else:
                    if x < x_points[midpoint + 1]:
                        last = midpoint - 1
                    else:
                        first = midpoint + 1

        return g


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10 ,10,100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(mean_err)

    # def test_with_poly_restrict(self):
    #     ass1 = Assignment1()
    #     a = np.random.randn(5)
    #     f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
    #     ff = ass1.interpolate(f, 1, 10, 10)
    #     xs = np.random.random(20)
    #     for x in xs:
    #         yy = ff(x)

if __name__ == "__main__":
    unittest.main()

