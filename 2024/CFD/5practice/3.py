import numpy as np
import matplotlib.pyplot as plt
import math
dt=0.01
n=10
dx=math.pi/n
l=[]
for i in range(101):
    l.append(i*dt)

U=np.zeros((101,11))
for j in range (0,n):
   
    U[1][j]=dt*((j*dx)**3)*(3*(j*dx)-(4*math.pi))
    
b=False
for i in range(1,100):
    U[i][0]=U[i][1]   #order question
    for j in range(1,n):

        U[i+1][j]=4*(dt**2/(dx**2))*(U[i][j+1]-2*U[i][j]+U[i][j-1])-U[i-1][j]+2*U[i][j]
    
    U[i][n]=U[i][n-1]
print(U)    