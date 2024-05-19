import numpy as np
import math
import matplotlib.pyplot as plt 
Pi=math.pi
interval = 1
n=10
dx=1/(n-1)

X = np.linspace(0,interval,n) 
A=np.full(n,-1/(12*dx*dx))
B=np.full(n,16/(12*dx**2))
C=np.full(n,-30/(12*dx**2))
D=np.full(n,16/(12*dx**2))
E=np.full(n,-1/(12*dx*dx))
H=np.zeros(n)
# My function
for i in range(n):
    x=dx*i
    H[i]=(-(dx*i)**2 * np.cos(2*Pi*dx*i))
a_0=np.zeros(n)
b_0=np.zeros(n)
y_0=np.zeros(n)
a_0[2]=(-B[1]-D[1]*b_0[1])/(C[1]+D[1]*a_0[1])
b_0[2]=A[1]/(-C[1]-D[1]*a_0[1])
y_0[2]=(H[1]-D[1]*y_0[1])/(C[1]+D[1]*a_0[1])

for i in range(2,n-1):
    a_0[i+1]=(-(B[i]+D[i]*b_0[i]+E[i]*b_0[i]*a_0[i-1])/(C[i]+D[i]*a_0[i]+E[i]*a_0[i]*a_0[i-1]+E[i]*b_0[i-1]))
    b_0[i+1]=(-(A[i])/(C[i]+D[i]*a_0[i]+E[i]*a_0[i]*a_0[i-1]+E[i]*b_0[i-1]))
    y_0[i+1]=((H[i]-D[i]*y_0[i]-E[i]*y_0[i]*a_0[i-1]-E[i]*y_0[i-1])/(C[i]+D[i]*a_0[i]+E[i]*a_0[i]*a_0[i-1]+E[i]*b_0[i-1]))

P_N=np.zeros(n)
P_N[n-1]=1
P_N[n-2]=(a_0[n-1]*P_N[n-1]+y_0[n-1])
for i in range(n-3,0,-1):
    P_N[i]=(a_0[i+1]*P_N[i+1]+b_0[i+1]*P_N[i+2]+y_0[i+1])
#Analytical solution for thomas
P=np.cos(2*Pi*X)*(X**2/(4*(Pi**2))-1/(8*(Pi**4)))-np.sin(2*Pi*X)*X/(2*(Pi**3))+X*(1-1/(4*(Pi**2)))+1/(8*(Pi**4))

print('error max_Thomas_alg difference: ',max(abs(P-P_N)))
plt.figure(figsize=(10, 5), layout='constrained')
plt.plot(X,P_N, label='Numerical_Thomas_5') 
plt.plot(X,P, label='Analytical') 

plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()


# Five-diagonal matrix method is more accurate because we have better aproximation of 1_D poisson equation with O(h^4). 
# we see that as 5-d matrix method gets closer to second boundary my error increases, and for big N its error stays same 0.069 - 0.077.
# Maybe that's some issue in code.
