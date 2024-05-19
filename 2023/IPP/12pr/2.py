import numpy as np
import matplotlib.pyplot as plt
import time
t=8
N=5 # for y
dy=1/100
n=5
dx=1/100

U=np.zeros((t+1,n+1,N+1),dtype='float64')
b=1
U[0]=np.full((n+1,N+1),8)

i=1
while b>10**-2 :
    if i>t-1:
            print("not steady")
            break     
    for k in range(1,n):
        U[i][0][k]=1
        U[i][n][k]=0
        for j in range(1,n):
            U[i][j][0]=0
            U[i][j][n]=0    
            U[i+1][j][k]=((dx*dy)/(2*dy+2*dx))*((dx**2)*(U[i][j][k+1]+U[i][j][k-1])+(dy**2)*(U[i][j-1][k]+U[i][j+1][k]))
            i+=1
    # b=max(abs(U[i]-U[i-18])) ?????
print(U)