import numpy as np
import matplotlib.pyplot as plt
import os
T=150
dt=0.003
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

x=np.linspace(0.0, 65,N+1)
y=np.linspace(0.0, 7,M+1)
X,Y=np.meshgrid(x,y)
U=np.zeros_like(X)
V=np.zeros_like(X)
P=np.zeros_like(X)
U_star=np.zeros_like(X)
V_star=np.zeros_like(X)
P_star=np.zeros_like(X)

U_next=np.zeros_like(X)
V_next=np.zeros_like(X)
P_next=np.zeros_like(X)

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
    b=int(0.5/dy)
    a=int(0.3/dy)
    print(dt,t)
    U_dx=central_diff_x(U)
    U_dy=central_diff_y(U)
    V_dx=central_diff_x(V)
    V_dy=central_diff_y(V)
    P_star_dx=central_diff_x(P_next)
    P_star_dy=central_diff_y(P_next)

    lap_U=laplace(U)
    lap_V=laplace(V)
    U_star=(U+dt*(-(U*U_dx+V*U_dy)+meu*lap_U-1/rho * P_star_dx))
    V_star=(V+dt*(-(U*V_dx+V*V_dy)+meu*lap_V-1/rho * P_star_dy))
    U_star[:b,:a]=0
    V_star[:b,:a]=0
    U_star[:b,0]=0
    U_star[b:,0]=1
   
    U_star[0,a:]=0
    U_star[-1,:]=0
    U_star[:,-1]=U_star[:,-2]
    U_star[b,:a]=0
    U_star[:b,a]=0

    V_star[b:,0]=V_star[b:,1]
    V_star[:b,0]=0
    V_star[0,a:]=0
    V_star[:,-1]=V_star[:,-2]
    V_star[-1,:]=0
    V_star[b,:a]=0
    V_star[:b,a]=0
   
    U_star_dx=central_diff_x(U_star)
    V_star_dy=central_diff_y(V_star)
    rhs= ( rho/dt*(U_star_dx+V_star_dy))
    stead=1
    const=0
    while const<=1 and  stead:
        pn = np.zeros_like(P)
        pn[1:-1,1:-1]=1/4 * (+P[1:-1,0:-2]+P[0:-2,1:-1]+P[1:-1,2:]+P[2:,1:-1]-dx**2*rhs[1:-1,1:-1])    
        pn [:b,:a]=0

        pn[:,-1]=pn[:,-2]
        pn[0,a:]=pn[1,a:]
        pn[:b,0]=0
        pn[b:,0]=pn[b:,1]
        pn[-1,:]=pn[-2,:]
        pn[b,:a]=pn[b+1,:a]
        pn[:b,a]=pn[:b,a+1]
        P=pn
        const += 0.02
        stead=subs(pn,P)    
    P_next=P_star+0.8*P
    pn_dx=central_diff_x(P)
    pn_dy=central_diff_y(P)
    U_next=( U_star - dt/rho * pn_dx)
    V_next=( V_star - dt/rho * pn_dy)
    
    U_next[:b,:a]=0
    V_next[:b,:a]=0
    U_next[:b,0]=0
    U_next[b:,0]=1
    V_next[b:,0]=V_next[b:,1]
    V_next[:b,0]=0
    U_next[0,a:]=0
    V_next[0,a:]=0
    U_next[:,-1]=U_next[:,-2]
    V_next[:,-1]=V_next[:,-2]
    U_next[-1,:]=0
    V_next[-1,:]=0
    U_next[b,:a]=0
    U_next[:b,a]=0
    V_next[b,:a]=0
    V_next[:b,a]=0

    if t>50:
        Steady=subs(U_next,U)
    t+=1
    U=0.8*U_next+(1-0.8)*U
    V=0.8*V_next+(1-0.8)*V
    # P_star=P_next
    # U=U_next
    # V=V_next
    # P=P_next
    plt.contourf(X,Y,P,levels=10,vmin=0,vmax=1.6)
    index_cut_x = int(N/10)
    index_cut_y = int(M/10)

    # plt.quiver(X, Y, U, V)
    plt.streamplot(X[::index_cut_y,::index_cut_x], Y[::index_cut_y,::index_cut_x], U[::index_cut_y,::index_cut_x], V[::index_cut_y,::index_cut_x], color='k')

    plt.colorbar()
    plt.draw()
    plt.pause(0.05)
    plt.clf()
index_cut_x = int(N/10)
index_cut_y = int(M/10)
plt.contourf(X,Y,V, alpha=0.5, cmap="plasma")
plt.title(f"Contour of V and Velocity direction at iteration {t}")
plt.colorbar()
plt.quiver(X, Y, U, V)
plt.plot(5+U[:,5],Y[:,5],color='black',linewidth=3)
plt.plot(20+U[:,20],Y[:,20],color='black',linewidth=3)
plt.plot(40+U[:,40],Y[:,40],color='black',linewidth=3)
plt.plot(50+U[:,-2],Y[:,-2],color='black',linewidth=3)

# plt.streamplot(X[::index_cut_y,::index_cut_x], Y[::index_cut_y,::index_cut_x], U[::index_cut_y,::index_cut_x], V[::index_cut_y,::index_cut_x], color='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
