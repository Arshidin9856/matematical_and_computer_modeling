import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


t=10000
dt=8/100
n=100
dx=8/100
v=dt/dx


l=np.zeros((t+1,n+1),dtype="float64")
y=[]
for i in range(t+1):
    y.append(i*dt)
x=[]
for i in range(n+1):
    x.append(i*dx)
b=1
i=0

l[t][0]=1
while b>10**-5:
        
    l[i][0]=1


    for j in range(1,n):
        if i ==0:
            l[0][j]=0
        l[i+1][j]=(l[i][j+1]+l[i][j-1])/2-(dt/(4*dx))*(l[i][j+1]**2-l[i][j-1]**2)

    l[i][n]=0
    i+=1
    b=max(abs(l[i]-l[i-1]))

print(l,b)   
x_mesh, t_mesh = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, t_mesh, l, cmap='viridis')

ax.set_xlabel('X-axis')
ax.set_ylabel('Time (t)')
ax.set_zlabel('f(x, t)')

plt.show()