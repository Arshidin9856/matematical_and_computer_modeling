import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

L=1.0
dx=0.1
dy=dx
dz=dx
x_size=int(L/dx)+1
y_size=int(L/dx)+1
z_size=int(L/dx)+1
X=np.linspace(0,L,x_size)
Y=np.linspace(0,L,y_size)
Z=np.linspace(0,L,z_size)
N=100
a=np.zeros((N+1,x_size,y_size))
P=np.zeros((x_size,y_size,z_size))
alpha=np.zeros((x_size,y_size))
Alpha=list()
Beta=np.zeros((x_size,y_size))

A=np.zeros((x_size,y_size))
B=np.zeros((x_size,y_size))
C=np.zeros((x_size,y_size))
d=np.zeros((x_size,y_size))

np.fill_diagonal(A, 1/dx**2)
np.fill_diagonal(C, 1/dx**2)




Alpha.append(np.zeros((x_size,y_size)))
Alpha.append(np.zeros((x_size,y_size)))
for i in range(x_size):
        for n in range(N+1):
            np.fill_diagonal(B,((2/dx**2)*np.cos(np.pi*n/N)-(3*(2/dx**2))))
            for j in range(0,x_size-1):
                for k in range(j-1,j):
                    B[j][k]=(1/dy**2)
                for k in range(j+1,j+2):
                    B[j][k]=(1/dy**2)
                B[0][y_size-1]=0.0
                B[x_size-1][y_size-2]=(1/dy**2)
            if i<=int(x_size/3) :
                for j in range(int(y_size/3),2*int(y_size/3)):
                    d[j][z_size-1]=-1/(dy**2)
                    #a[n][j][z_size-1]=1.0
                    #Beta[]
            if i==0:
                #for j in range(int(y_size/3)):
                #    d[j][0]=-1/(dy**2)
                for k in range(int(z_size/3)):
                   Beta[0][k]=275
                    # d[k][0]=-1/(dy**2)

            for j in range(0,x_size-1):
                alpha=-np.matmul(np.linalg.inv(B+np.matmul(C,Alpha[j])),A)
                Alpha.append(alpha)
                Beta[j+1]=np.matmul(np.linalg.inv(B+np.matmul(C,Alpha[j])),d[j]-np.matmul(C,(Beta[j])))  

            for j in range(x_size-1,-1,-1):
                a[n][j-1]=np.transpose(np.add(np.matmul((Alpha[j]),(a[n][j])),(Beta[j])))
        
            for j in range(y_size):
                for k in range(z_size):
                    P[i,j,k]+=np.sin(np.pi*i*n/N)*a[n,j,k]  

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
print(P)
graph(P*10)



with open('res.dat', 'w', newline='') as file:
    writer = csv.writer(file)
    file.write("VARIABLES= ")
    file.write('"X"')
    file.write(", ")
    file.write('"Y"')
    file.write(", ")
    file.write('"Z"')
    file.write(", ")
    file.write('"U"')
    file.write("\n")
    for i in range(x_size):
        for j in range(y_size):
            for k in range(z_size):
                file.write(str(i*dx))
                file.write(" ")
                file.write(str(j*dy))
                file.write(" ")
                file.write(str(k*dz))
                file.write(" ")
                file.write(str(P[i,j,k]))
                file.write("\n")
with open('res1.dat', 'w', newline='') as file:
    writer = csv.writer(file)
    file.write("VARIABLES= ")
    file.write('"Y"')
    file.write(", ")
    file.write('"Z"')
    file.write(", ")
    file.write('"U"')
    file.write("\n")
    #for i in range(x_size):
    for j in range(y_size):
        for k in range(z_size):
       #         if j==5:
                    file.write(str(j*dx))
                    file.write(" ")
                    file.write(str(k*dz))
                    file.write(" ")
                    file.write(str(P[1,j,k]))
                    file.write("\n")