import time
import random
from functionUtils import *
from sampleFunctions import *
from sklearn.cluster import KMeans
import numpy as np
import matplotlib



class MyShape(AbstractShape):
    def __init__(self, area: np.ndarray):
        self._area = area

    def area(self) -> np.float32:
        return self._area


class Assignment5:
    def __init__(self):
        pass


    def area(self,contour: callable, maxerr=0.001)->np.float32:
        all_points = contour(3000)
        y_list = [y for x, y in all_points]
        x_list = [x for x, y in all_points]
        a1, a2 = 0., 0.
        x_list.append(x_list[0])
        y_list.append(y_list[0])
        for j in range(len(x_list) - 1):
            a1 += np.float64(x_list[j]) * np.float64(y_list[j + 1])
            a2 += np.float64(y_list[j]) * np.float64(x_list[j + 1])
        l = np.float64(abs(a1 - a2) / 2.)
        return l


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        all_points = []
        for i in range(1000):
            x,y = sample()
            while [x,y] in all_points:
                x, y = sample()
            all_points.append([x,y])

        p_t = all_points[0]
        all_points.sort(key=lambda p: (p[0] - p_t[0]) ** 2 + (p[1] - p_t[1]) ** 2)
        kmeans = KMeans(n_clusters=25).fit(all_points).cluster_centers_.tolist()
        kmeans.sort(key=lambda x: x[0])
        start_p = kmeans[0]
        c_p = [start_p]

        if len(kmeans) == 2:
            c_p.append(kmeans[1])
            c_p.append(c_p[0])
        else:
            for i in range(len(kmeans) - 1):
                kmeans.sort(key=lambda p: (p[0] - start_p[0]) ** 2 + (p[1] - start_p[1]) ** 2)
                c = kmeans[1]
                kmeans.remove(start_p)
                start_p = c
                c_p.append(c)

        y_list = [y for x, y in c_p]
        x_list = [x for x, y in c_p]
        a1, a2 = 0., 0.
        x_list.append(x_list[0])
        y_list.append(y_list[0])
        for j in range(len(x_list) - 1):
            a1 += np.float64(x_list[j]) * np.float64(y_list[j + 1])
            a2 += np.float64(y_list[j]) * np.float64(x_list[j + 1])
        l = np.float64(abs(a1 - a2) / 2.)

        return MyShape(l)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        circ = Circle(1,1,1,1).sample
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        print(shape)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
