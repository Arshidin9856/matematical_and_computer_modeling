from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
#Fourie method:  boundary conditions for matrix projection doesnt work as it should. I dont see clear area at the bottom instead its some values at the z direction 
n=11
if n%3==0:
    raise IndexError
# boundary
a_x=n//3
b_x=round(n/3)+n//3+1
print(a_x,b_x)
# coeff
dx=1/(n-1)
dy=1/(n-1)
dz=1/(n-1)
eps=10**-5
#matrices
A=np.eye(n)/dx**2
C=np.eye(n)/dx**2
U_prev=np.zeros((n,n,n))
U_n=np.zeros((n,n,n))
D=np.zeros(n)

a=np.zeros((n,n,n))

def compute_P_ijk(a_njk, N):
        i, j, k = np.indices(a_njk.shape)
        index = np.arange(1, N+1)
        sin_term = np.sin(np.pi * i[..., np.newaxis] * index / N)
        return np.sum(a_njk * sin_term, axis=-1)
for k in range(n):
    # count n times for every k layer !!!!!!!!!!
    for count in range(n):
# First step   find a_n then find U
        B=np.eye(n)*(2/dx**2+2/dy**2 + 2*np.cos(np.pi*count/n)/dz**2) + np.diag(np.ones(n-1)*(-1/dy**2), 1) + np.diag(np.ones(n-1)*(-1/dy**2), -1) 
        if count >a_x and count<=b_x:
            betha1=[[0]*n,[0]*a_x+[275]*(n-a_x)]
        elif count>b_x:
            betha1=[[0]*n,[0]*n] # column
            D[0]=(-1/dy**2) 
            # betha1=[[0]*n,[500]*a_x+[500]*(b_x-a_x)+[500]*(n-b_x)]
        else:
            betha1=[[0]*n,[0]*n] # column
        print(dy)
        alpha1=[[[0]*n]*n,[[0]*n]*n] # matrix
        for _ in range(1,n):
            bettha_temp=np.dot(np.linalg.inv(-(B+np.dot(C,alpha1[-1]))),D+betha1[-1])
            alpha_temp=np.dot(np.linalg.inv(-(B+np.dot(C,alpha1[-1]))),A)
            alpha1.append(alpha_temp.tolist())
            betha1.append(bettha_temp.tolist())
        p_n1=[[0]*n,[0]*n] # column
 
        for j in range(n-1,0,-1):
            p_n1.append((np.dot(alpha1[j],p_n1[-1])+betha1[j]).tolist())
        temp=np.array(p_n1[1:])

        a[count]=temp[::-1]    
        for i in range(n):
                for j in range(n):
                    U_n[i,j,k]+=np.sin(np.pi*k*count/n)*a[i,j,count]  
U_n_func=compute_P_ijk(a,n)
# print((U_n==U_n_func).all())
def graph(T, n=n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    fig = go.Figure(data=[go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=T.flatten(),
        isomin=T.min(),
        isomax=T.max(),
        opacity=0.1,
        surface_count=21,
        colorscale='Viridis')])
    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')))
    fig.show()
graph(U_n)
graph(U_n_func)

