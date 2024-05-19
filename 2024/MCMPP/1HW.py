import numpy as np
import math 
import matplotlib.pyplot as plt 
Pi=math.pi
interval = 1
dx=0.01
n=int(interval/dx)+1
n=11
X = np.linspace(0,interval,n) 
# analytical solution
# P=np.cos(2*Pi*X)*(X**2/(4*(Pi**2))-1/(8*(Pi**4)))-np.sin(2*Pi*X)*X/(2*(Pi**3))+X*(1-1/(4*(Pi**2)))+1/(8*(Pi**4))
# P(0)=0
# P(1)=1
A=C=1/dx**2
B=-2/dx**2
a_0=[0]
b_0=[0]
P_N=[1]
D=[]
for i in range(n):
    D.append(-(dx*i)**2 * np.cos(2*Pi*dx*i))
print(D)
for i in range(1,n):
    
    a_0.append(-A/(B+C*a_0[-1]))
    b_0.append((D[i]-C*b_0[-1])/(B+C*a_0[-1]))

for i in range(n-1,0,-1):
    P_N.append(a_0[i]*P_N[-1]+b_0[i])
# max_error=max(abs(P-P_N[::-1]))
# print(f'{max_error} - max error\n{n} - iterations')    
# for i in range(n):
    # if abs(P[i]-P_N[::-1][i])>=max_error: print(P[i],P_N[::-1][i], f'max error at x {i*dx}' )  
plt.figure(figsize=(5, 2.7), layout='constrained')
# plt.plot(X, P, label='Analytical')  # Plot some data on the (implicit) axes.
plt.plot(X, P_N[::-1], label='Numerical')  # Plot some data on the (implicit) axes.
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()


# For x between 0 and 1 PDE looks like straight line. 
# Tridiagonal method looks simple in our case because we have constant A B C and its always stable.
# About equation it allows us predict some value along a road or pipe 