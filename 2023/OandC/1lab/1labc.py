import numpy as np
from scipy.optimize import linprog
def simplex_method(c, A, b):
    # Решаем задачу линейного программирования с помощью симплекс-метода
    result = linprog(c, A_ub=A, b_ub=b, method='simplex')

    return result.x, result.fun
c = np.array([-1, -2])

# Задаем матрицу A и вектор b
A = np.array([[1, 1],
              [3, 2]])
b = np.array([6, 12])


A = np.array([[10, 2,1], 
              [7, 3,2], 
              [2,4,1]])
b = np.array([100, 72,80])
c= np.array([ -22,-6,-2])

# Решаем задачу с помощью симплекс-метода

#1
n=2
m=3

Conditions = np.array([[3, 2], 
                       [3, 5], 
                       [5,6]])
b = np.array([600, 800,1100])
max_f = np.array([ -30,-40])

x, f = simplex_method(max_f, Conditions, b)

print(f"Оптимальное решение: x = {x}, f = {-f}")

# 2
# n=3
# m=3

# Conditions = np.array([[10, 2,1], 
#                        [7, 3,2], 
#                        [2,4,1]])
# b = np.array([0,100, 72,80],float)
# max_f = np.array([ -22,-6,-2])

