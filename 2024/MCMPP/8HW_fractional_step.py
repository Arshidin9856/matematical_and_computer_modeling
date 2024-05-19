import numpy as np
import matplotlib.pyplot as plt
n=26
if n%3==0:
    raise IndexError
# boundary
a_x=n//3
b_x=round(n/3)+n//3+1
# coeff
dx=1/(n-1)
dy=1/(n-1)
dt=dx*dx/4
eps=10**-5
A=-1/(2*dx**2)
C=-1/(2*dx**2)
B=1/dx**2+1/dt
#matrices
U_prev=np.zeros((n,n))
U_n=np.zeros((n,n))
D=np.zeros((n,n))

fig,axis=plt.subplots()
pcm=axis.pcolormesh(U_n,cmap=plt.cm.jet,vmin=0,vmax=275)
plt.colorbar(pcm,ax=axis)

iter_thomas=0
max_d=4
while   max_d> eps and iter_thomas<10000:
# First step    
    for i in range(1,n-1):
        for j in range(1,n-1):
                D[i][j]=U_prev[i][j]/dt + 0.5*(U_n[i+1][j]-2*U_n[i][j]+U_n[i-1][j])/(dx**2)+(U_prev[i][j+1]-2*U_prev[i][j]+U_prev[i][j-1])/(dy**2)
    for j in range(1,n-1):
        if j >a_x and j<=b_x:
            p_n1=[0,275]
        else:     
            p_n1=[0,0]
        betha1=[0,0]
        alpha1=[0,0]
        for i in range(1,n):
            alpha1.append(-A/(B+C*alpha1[-1]))
            betha1.append((D[i][j]-C*betha1[-1])/(B+C*alpha1[-1]))
        for i in range(n-1,0,-1):
            p_n1.append(alpha1[i]*p_n1[-1]+betha1[i])
        temp=np.array(p_n1[1:])
        
        U_n.T[j]=temp[::-1]    
## second step
    for i in range(1,n-1):
        for j in range(1,n-1):
                D[i][j]=U_n[i][j]/dt - 0.5 * (U_prev[i][j+1]-2*U_prev[i][j]+U_prev[i][j-1])/(dy**2)
    for i in range(1,n-1):
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
# exit condition
    max_d=0
    for i in range(n):
        for j in range(n):
            if max_d<abs(U_n[i][j]-U_prev[i][j]):
                max_d=abs(U_n[i][j]-U_prev[i][j])
    # if max_d>0.008:
    #     pcm.set_array(U_n)
    #     axis.set_title(f'at iter  {iter_thomas}')
    #     plt.pause(0.001)
    for i in range(n):
        for j in range(n):
            U_prev[i][j]=U_n[i][j]   
    iter_thomas+=1
    print(iter_thomas, max_d)

pcm.set_array(U_n)
axis.set_title(f'at iter  {iter_thomas}')
plt.show()
# ALt_res=np.load('res_alt_dir.npy')
# ALt_iter=np.load('iter.npy')
# # print(ALt_res-U_n)
# print('Max difference: ',np.absolute(ALt_res-U_n).max())
# print('num of iter(fractional step): ',iter_thomas)
# print('num of iter(alternating direction): ',ALt_iter)
