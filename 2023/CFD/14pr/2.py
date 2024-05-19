import numpy as np
import matplotlib.pyplot as plt
import os
T=100000
dt=0.001
N=40
M=40
dx=1/N
dy=1/M
meu=0.01 #const same as RE=100 Re=1/meu
rho=1#const
Re=100

t=0

def subs(L_i,L_j):
    
    for i in range(N):
        for j in range(M):
            if abs(L_i[i][j]-L_j[i][j])>10**-6:
                return True     
    print('Steady')
    return False

x=np.linspace(0.0, 1,N+1)
y=np.linspace(0.0, 1,M+1)
X,Y=np.meshgrid(x,y)
U=np.zeros_like(X)
V=np.zeros_like(X)
P=np.zeros_like(X)
def central_diff_x(f):
    diff = np.zeros_like(f)
    diff [1:-1,1:-1] = (f[1:-1,2:]-f[1:-1,0:-2])/(2*dx)
    return diff
def central_diff_y(f):
    diff = np.zeros_like(f)
    diff [1:-1,1:-1] = (f[2:,1:-1]-f[0:-2,1:-1])/(2*dy)
    return diff
def laplace(f):
    diff = np.zeros_like(f)
    diff [1:-1,1:-1] = (f[1:-1,0:-2]+f[0:-2,1:-1]-4*f[1:-1,1:-1]+f[1:-1,2:]+f[2:,1:-1])/(dy**2)
    return diff


Steady=1
while Steady and t<T:
    print(dt,t)

    U_dx=central_diff_x(U)
    U_dy=central_diff_y(U)
    V_dx=central_diff_x(V)
    V_dy=central_diff_y(V)
    lap_U=laplace(U)
    lap_V=laplace(V)
    U_star=(U+dt*(-(U*U_dx+V*U_dy)+meu*lap_U))
    V_star=(V+dt*(-(U*V_dx+V*V_dy)+meu*lap_V))



    # for j in range(1,N):
    #     for k in range(1,M):
    #         U_star[j][k]=(dt/(Re*dx*dx))*(U[j+1][k]-4*U[j][k]+U[j-1][k]+U[j][k+1]+U[j][k-1])-dt*(U[j][k]*(U[j+1][k]-U[j-1][k])/(dx*2)+V[j][k]*(U[j][k+1]-U[j][k-1])/(dy*2)) + U[j][k]
    #         V_star[j][k]=(dt/(Re*dx*dx))*(V[j+1][k]-4*V[j][k]+V[j-1][k]+V[j][k+1]+V[j][k-1])-dt*(U[j][k]*(V[j+1][k]-V[j-1][k])/(dx*2)+V[j][k]*(V[j][k+1]-V[j][k-1])/(dy*2)) + V[j][k]

    # at walls
    # for j in range(0,M+1):
    #     #inlet, outlet top
    #      #bottom wall
    #     U[j][-1]=0
    #     V[j][-1]=0
    #     U[j][0]=0
    #     V[j][0]=0
    # for i in range(0,N+1):
    #     #bottom wall
    #     U[-1][i]=1
    #     V[-1][i]=0
    #     U[0][i]=0
    #     V[0][i]=0
    
    U_star[:,0]=0
    U_star[0,:]=0
    U_star[:,-1]=0
    U_star[-1,:]=1
    V_star[:,0]=0
    V_star[0,:]=0
    V_star[:,-1]=0
    V_star[-1,:]=0

    U_star_dx=central_diff_x(U_star)
    V_star_dy=central_diff_y(V_star)
    rhs= ( rho/dt*(U_star_dx+V_star_dy))
    stead=1
    const=0
    while const<=1 and  stead:
        pn = np.zeros_like(P)
        pn[1:-1,1:-1]=1/4 * (+P[1:-1,0:-2]+P[0:-2,1:-1]+P[1:-1,2:]+P[2:,1:-1]-dx**2*rhs[1:-1,1:-1])    
        # for i in range(1,N):
        #     for j in range(1,M):
        #         pn[i][j]=1/4*(P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]-(rho*dy)/(dt*2)*((U_star[i+1][j]-U_star[i-1][j]+V_star[i][j+1]-V_star[i][j-1])))
        pn[:,-1]=pn[:,-2]
        pn[0,:]=pn[1,:]
        pn[:,0]=pn[:,1]
        pn[-1,:]=0
        # for j in range(0,M+1):
        #     #inlet, outlet top
            
        #     P[j][0] = P[j][1]
        #     P[j][-1] = P[j][-2]
            
        # for i in range(0,N+1):
        #     #bottom wall
        #     P[0][i] = P[1][i]
        #     #top wall
        #     P[-1][i] =0
        # outlet 
        P=pn
        const += 0.02
        stead=subs(pn,P)    
    # for j in range(1,N):
    #     for k in range(1,M):
    #         # inlet
    #         U_next[j][k]=U_star[j][k]-dt/rho *(-P[j-1][k]+P[j+1][k]/(dx*2))
    #         V_next[j][k]=V_star[j][k]-dt/rho *(-P[j][k-1]+P[j][k+1]/(dy*2))
    pn_dx=central_diff_x(pn)
    pn_dy=central_diff_y(pn)
    U_next=( U_star - dt/rho * pn_dx)
    V_next=( V_star - dt/rho * pn_dy)

    # for j in range(0,M+1):
    #     #inlet, outlet top
    #      #bottom wall
    #     U_next[j][-1]=0
    #     V_next[j][-1]=0
    #     U_next[j][0]=0
    #     V_next[j][0]=0
    # for i in range(0,N+1):
    #     #bottom wall
    #     U_next[-1][i]=1
    #     V_next[-1][i]=0
    #     U_next[0][i]=0
    #     V_next[0][i]=0
    
    U_next[:,0]=0
    V_next[:,0]=0
    
    U_next[0,:]=0
    V_next[0,:]=0
    
    U_next[:,-1]=0
    V_next[:,-1]=0

    U_next[-1,:]=1
    V_next[-1,:]=0
    if t>50:
        Steady=subs(U_next,U)
    t+=1
    U=U_next
    V=V_next
    P=pn
print(P)








print(f'{N,M} - N,M')
index_cut_x = int(N/10)
index_cut_y = int(M/10)
plt.contourf(X,Y,P, alpha=0.5, cmap="plasma")
plt.title("Contour of Velocity U and Velocity direction")
plt.colorbar()
plt.quiver(X, Y, U, V)
# plt.streamplot(X[::index_cut_y,::index_cut_x], Y[::index_cut_y,::index_cut_x], U[::index_cut_y,::index_cut_x], V[::index_cut_y,::index_cut_x], color='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
y = np.linspace(-5, 5,M+1)

plt.plot(U_next[N//2,:],Y, label=f'x = {N//2}')
plt.xlabel('y')
plt.ylabel('f(x, y)')
plt.title(f'Values of f(x, y) at x = {N//2}')
plt.legend()
plt.grid(True)
plt.show()
#print(un)
#print(us)
#print(vs)