import numpy as np
import math 
import matplotlib.pyplot as plt 
Pi=math.pi
interval = 1
dx=0.01
dt=0.5*dx**2
n=int(interval/dx)+1
eps=0.1
# loop for increasing accuracy 
for i in range(5):    
    X = np.linspace(0,interval,n) 
    # U(0)=0
    # U(1)=0
    # T(0)=(3x**5-5x**4+2x)
    A=C=-2/dx**2
    B=4/dx**2+1/dt
    D=np.zeros(n)
    U_prev=np.empty(n)
    # initial values of U
    for i in range(n):
        x=dx*i
        U_prev[i]=(3*(x**5)-5*(x**4)+2*x)
    iter=0
    while True:
        a_0=[0]
        b_0=[0]
        P_N=[0]

        for i in range(n):
            D=U_prev/dt
        for i in range(1,n):
            a_0.append(-A/(B+C*a_0[-1]))
            b_0.append((D[i]-C*b_0[-1])/(B+C*a_0[-1]))

        for i in range(n-1,0,-1):
            P_N.append(a_0[i]*P_N[-1]+b_0[i])
        U_Next=np.array(P_N)
        
        if max(abs(U_Next-U_prev))<eps:
        
            break
        iter+=1
        U_prev=U_Next
    print(iter)
    # analytical solution
    P=np.empty(n)
    for i in range(n):
            res=0
            for ind in range (1,101):
                res+=(240*(2*((-1)**(ind+1))-1)/((Pi*ind)**5))*math.sin(Pi*ind*i*dx)*math.exp(-2*((Pi*ind)**2)*dt*iter)
            P[i]=res
    print(eps,'error max: ',max(abs(P-U_prev[::-1])))
    plt.figure(figsize=(10, 5), layout='constrained')
    plt.plot(X, U_prev[::-1], label='Numerical') 
    plt.plot(X, P, label='Analytical') 
    plt.xlabel(f'x label: error max: ,{max(abs(P-U_prev[::-1]))}')
    plt.ylabel('y label')
    plt.title(f"Simple Plot:  epsilon = {eps} , iter: {iter}")
    plt.legend()
    plt.show()

    eps*=0.1

# As we see increasing the number of iterations and accuracy values com closer to boundary values from initial. 
# But increasing accuracy results in big amount of iterations requaried.
# And also in numerical solution first value should be zero, but my code doesnt give me left boundary.
