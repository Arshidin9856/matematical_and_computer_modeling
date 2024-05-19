import numpy as np
import matplotlib.pyplot as plt

# U[0][x]=1-x
# U[t][1]=0
# U[t][0]=1                 Stability: c*dt/dx<=1
n=11
dx=1/(n-1)
c=-0.5
dt=-0.01*dx/(2*c)
X=np.linspace(0,1,n)
def agains_flow(iter_num):
    U_prev=1-X
    U_new=np.zeros(n)
    i=0
    b=1
    while i<iter_num: 
        i+=1
        U_prev[0]=1
        U_prev[n-1]=0

        for j in range(1,n-1):
                
            U_new[j]=U_prev[j]-(U_prev[j+1]-U_prev[j])*c*dt/dx

        b=max(abs(U_new-U_prev))
        U_prev=U_new
        # print(U_new,U_prev)
    plt.figure(figsize=(10, 5), layout='constrained')
    plt.plot(X,U_new, label='Against flow') 
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title(f"Simple Plot for time {iter_num}")
    plt.legend()
    plt.show()
    return b
time_S=[100,1000,10000]
time_S=list(map(agains_flow,time_S))
print('max error for diff time: ', time_S)
