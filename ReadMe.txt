*Assignment 1:

The function will receive a function f, a range, and a number of points to use.
The function will return another “interpolated” function g. During testing, g will be called with various floats x to test for the interpolation errors. 

*Assignment 2:

The function will receive 2 functions- f_1, f_2, and a float maxerr.
The function will return an iterable of approximate intersection Xs, such that:
∀x∈X,|f_1 (x)-f_2 (x)|<maxerr
∀x_i x_j∈X,|x_i-x_j |>maxerr
 
*Assignment 3:

-Assignment3.integrate(…) receives a function f, a range, and a number of points n.
It must return approximation to the integral of the function f in the given range.
You may call f at most n times. 


-Assignment3.areabetween(..)  receives two functions f_1,f_2 .
It must return the area between f_1,f_2 . 
 
*Assignment 4:
The function will receive an input function that returns noisy results. The noise is normally distributed. 
Assignment4.fit should return a function g fitting the data sampled from the noisy function. Use least squares fitting such that g will exactly match the clean (not noisy) version of the given function. 
To aid in the fitting process the arguments a and b signify the range of the sampling. The argument d is the expected degree of a polynomial that would match the clean (not noisy) version of the given function. 
Additional parameter to Assignment4.fit is maxtime representing the maximum allowed runtime of the function.


*Assignment 5:
-Implement the function Assignment5.area(…).
The function will receive a shape contour and should return the approximate area of the shape.


-Implement the function Assignment5.fit_shape(…)  and the class MyShape following the pydoc instructions.
The function will receive a generator (a function that when called), will return a point (tuple) (x,y), a that is close to the shape contour.