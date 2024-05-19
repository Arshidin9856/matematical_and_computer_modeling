import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from celluloid import Camera
import numpy as np
#Alternating direction method: 
n=26
if n%3==0:
    raise IndexError
# boundary
a_x=n//3
b_x=round(n/3)+n//3+1
# coeff
dx=1/(n-1)
dy=1/(n-1)
dz=1/(n-1)
dt=dx*dx/4
eps=10**-5
A=-1/(2*dx**2)
C=-1/(2*dx**2)
B=1/dx**2+1/dt
#matrices
U_prev=np.zeros((n,n,n))
U_n=np.zeros((n,n,n))
D=np.zeros((n,n,n))


iter_thomas=0
max=4
while   max> eps and iter_thomas<1000:
# First step    
    for i in range(1,n-1):
        for j in range(1,n-1):
            for k in range(1,n-1): 
                D[i][j][k]=U_prev[i][j][k]/dt + 0.5 * (U_n[i+1][j][k]-2*U_n[i][j][k]+U_n[i-1][j][k])/(dx**2) + (U_prev[i][j+1][k]-2*U_prev[i][j][k]+U_prev[i][j-1][k])/(dy**2) + (U_prev[i][j][k+1]-2*U_prev[i][j][k]+U_prev[i][j][k-1])/(dz**2)
    for j in range(1,n-1):
        for k in range(1,n-1):
            if j >a_x and j<=b_x and k>a_x and k<=b_x:
                p_n1=[0,275]
            else:     
                p_n1=[0,0]
            betha1=[0,0]
            alpha1=[0,0]
            for i in range(1,n):
                alpha1.append(-A/(B+C*alpha1[-1]))
                betha1.append((D[i][j][k]-C*betha1[-1])/(B+C*alpha1[-1]))
            for i in range(n-1,0,-1):
                p_n1.append(alpha1[i]*p_n1[-1]+betha1[i])
            temp=np.array(p_n1[1:])
            
            U_n.T[j]=temp[::-1]    
## second step
    for i in range(1,n-1):
        for j in range(1,n-1):
            for k in range(1,n-1):
                
                D[i][j][k]=U_n[i][j][k]/dt - 0.5 * (U_n[i][j+1][k]-2*U_n[i][j][k]+U_n[i][j-1][k])/(dy**2) 
    for i in range(1,n-1):
        for k in range(1,n-1):

            if i >a_x and i<=b_x:
                betha2=[0,275]
            else:     
                betha2=[0,0]
            alpha2=[0,0]
            p_n2=[0,0]
            for j in range(1,n):
                alpha2.append(-A/(B+C*alpha2[-1]))
                betha2.append((D[i][j]-C*betha2[-1])/(B+C*alpha2[-1]))
            for j in range(n-1,0,-1):
                p_n2.append(alpha2[j]*p_n2[-1]+betha2[j])
            temp1=np.array(p_n2[1:])
            U_n[i]=temp1[::-1] 
## third step
    for i in range(1,n-1):
        for j in range(1,n-1):
            for k in range(1,n-1):
                
                D[i][j][k]=U_n[i][j][k]/dt - 0.5 * (U_n[i][j][k+1]-2*U_n[i][j][k]+U_n[i][j][k-1])/(dz**2) 
    for i in range(1,n-1):
        for k in range(1,n-1):

            if i >a_x and i<=b_x:
                betha2=[0,275]
            else:     
                betha2=[0,0]
            alpha2=[0,0]
            p_n2=[0,0]
            for j in range(1,n):
                alpha2.append(-A/(B+C*alpha2[-1]))
                betha2.append((D[i][j][k]-C*betha2[-1])/(B+C*alpha2[-1]))
            for j in range(n-1,0,-1):
                p_n2.append(alpha2[j]*p_n2[-1]+betha2[j])
            temp1=np.array(p_n2[1:])
            U_n[i]=temp1[::-1] 

# exit condition
    max=0
    for i in range(n):
        for j in range(n):
            if max<abs(U_n[i][j]-U_prev[i][j]):
                max=abs(U_n[i][j]-U_prev[i][j])
    # if max>0.008:
    #     pcm.set_array(U_n)
    #     axis.set_title(f'at iter  {iter_thomas}')
    #     plt.pause(0.001)
    for i in range(n):
        for j in range(n):
            U_prev[i][j]=U_n[i][j]   
    iter_thomas+=1
    print(iter_thomas, max)

plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
camera = Camera(fig)
    # ax.plot(x, y, z, color='blue')
    # camera.snap()
anim = camera.animate(blit=False, interval=10)
anim.save('3d.mp4')
# np.save('res_alt_dir',U_n)
# np.save('iter',iter_thomas)
