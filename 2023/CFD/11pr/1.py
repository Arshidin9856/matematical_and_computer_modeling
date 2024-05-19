import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


t=100
dt=4/t
n=100
dx=8/n
l=np.zeros((t+1,n+1))
y=[]
for i in range(t+1):
    y.append(i*dt)
x=[]
for i in range(n+1):
    x.append(i*dx)
b=True
i=0
def subs(L_i,L_j,iter):
    
    for i in range(n):
        if abs(L_i[i]-L_j[i])>10**-5:
             
          return True 
    print('Steady')
    return False
l[t][0]=1
while b:
    if i==t:

        break
    l[i][0]=1

    temp=np.full(n+1,1)
    for j in range(1,n):
        if i ==0:
            l[0][j]=0

        temp[j]=l[i][j]-(dt/dx)*(l[i][j+1]-l[i][j])
        l[i+1][j]=(l[i][j]+temp[j]-(dx/dt)*(temp[j]-temp[j-1]))/2

    l[i][n]=0
    i+=1
    b=subs(l[i],l[i-1],i)

print(l)   
x_mesh, t_mesh = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, t_mesh, l, cmap='viridis')

ax.set_xlabel('X-axis')
ax.set_ylabel('Time (t)')
ax.set_zlabel('f(x, t)')

plt.show()