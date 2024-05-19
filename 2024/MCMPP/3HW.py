import numpy as np
import math 
import matplotlib.pyplot as plt 
Pi=math.pi
interval = 1
dx=0.01
dt=0.5*dx**2
n=int(interval/dx)+1
eps=0.1*10**-5
print('Stability for Thomas: \n4/dx**2 < 4/dx**2 + 1/dt\nStability for Simple:\na*dt/dx**2 <= 1/2')
X = np.linspace(0,interval,n) 
print('DU/DT=aD**2U/DX**2\nU(0)=0, U(1)=0,T(0)=(3x**5-5x**4+2x)') 
A=C=-2/dx**2
B=4/dx**2+1/dt
D=np.zeros(n)
U_prev=np.empty(n)
# initial values of U
for i in range(n):
    x=dx*i
    U_prev[i]=(3*(x**5)-5*(x**4)+2*x)

iter_thomas=0
while True:
    a_0=[0,0]
    b_0=[0,0]
    P_N=[0,0]
    for i in range(n):
        D=U_prev/dt
    for i in range(1,n):
        a_0.append(-A/(B+C*a_0[-1]))
        b_0.append((D[i]-C*b_0[-1])/(B+C*a_0[-1]))
    for i in range(n-1,0,-1):
        P_N.append(a_0[i]*P_N[-1]+b_0[i])
    U_N=np.array(P_N[1:])
    if max(abs(U_N-U_prev))<eps:
        break
    iter_thomas+=1
    U_prev=U_N
#Analytical solution for thomas
P=np.empty(n)
for i in range(n):
        res=0
        for ind in range (1,101):
            res+=(240*(2*((-1)**(ind+1))-1)/((Pi*ind)**5))*math.sin(Pi*ind*i*dx)*math.exp(-2*((Pi*ind)**2)*dt*iter_thomas)
        P[i]=res

# SIMPLE
n=100
t=100000
dt=0.1*dx**2
dx=0.01
T_prev=np.zeros((t+1,n+1),dtype='float64')
for i in range(n+1):
    x=dx*i
    T_prev[0][i]=(3*(x**5)-5*(x**4)+2*x)
iter=0
b=1
def subs(L_i,L_j):
    for i in range(n-1):
            if abs(L_i[i]-L_j[i])>eps:
                return True     
    print('Steady')
    return False
j=0
while b:
    T_prev[j][0]=0
    T_prev[j][n]=0
    for i in range(1,n):
        T_prev[j+1][i]=(2*dt/dx**2)*(T_prev[j][i+1]-2*T_prev[j][i]+T_prev[j][i-1])+T_prev[j][i]
    
    b=subs(T_prev[j+1],T_prev[j])
    iter+=1
    j+=1
# analytical solution
P_simple=np.empty(n+1)
for i in range(n+1):
        res=0
        for ind in range (1,101):
            res+=(240*(2*((-1)**(ind+1))-1)/((Pi*ind)**5))*math.sin(Pi*ind*i*dx)*math.exp(-2*((Pi*ind)**2)*dt*iter)
        P_simple[i]=res
print('error max_Thomas_alg: ',max(abs(P-U_prev[::-1])))
print('error max_Simple: ',max(abs(P_simple-T_prev[iter])))
print('max_difference: ',max(abs(U_prev[::-1]-T_prev[iter])))

print('iter_requaired: Thomas = ',iter_thomas,'iter_requaired: Simple = ',iter)
plt.figure(figsize=(10, 5), layout='constrained')
plt.plot(X, U_prev[::-1], label='Numerical_Thomas') 
plt.plot(X, T_prev[iter], label='Numerical_Simple') 
plt.plot(X, P, label='Analytical_Thomas') 
plt.plot(X, P_simple, label='Analytical_simple') 

plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()


# As we see Thomas method is much faster for same accuracy epsilon. 
# But Simple method gives smaller error. We see correlation between number of iter and accuracy.
# And also in numerical solution of Simple method we see that analytical solution is more precize with analytical.
