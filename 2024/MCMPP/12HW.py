import numpy as np
import matplotlib.pyplot as plt
import os

T=150
N=40
M=40

dt=0.003
dx=1/N
dy=1/M
b=int(0.5/dx)
a=int(0.3/dy)
    

meu=0.01 #const same as RE=100 Re=1/meu
rho=1#const
Re=100

t=0
Steady=1

def subs(L_i,L_j):
    
    for i in range(N):
        for j in range(M):
            if abs(L_i[i][j]-L_j[i][j])>10**-6:
                return True     
    print('Steady')
    return False

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


x=np.linspace(0.0, 60,N+1)
y=np.linspace(0.0, 7,M+1)
X,Y=np.meshgrid(x,y)

U=np.zeros_like(X)
V=np.zeros_like(X)
P=np.zeros_like(X)

U_star=np.zeros_like(X)
V_star=np.zeros_like(X)

U_next=np.zeros_like(X)
V_next=np.zeros_like(X)

while Steady and t<T:
    U_dx=central_diff_x(U)
    U_dy=central_diff_y(U)
    V_dx=central_diff_x(V)
    V_dy=central_diff_y(V)
    lap_U=laplace(U)
    lap_V=laplace(V)
    U_star=(U+dt*(-(U*U_dx+V*U_dy)+meu*lap_U))
    V_star=(V+dt*(-(U*V_dx+V*V_dy)+meu*lap_V))
    # initial for U,V at left wall 
    U_star[:a,0]=0
    U_star[a:b,0]=1
    U_star[b:,0]=0
    V_star[:,0]=0

    # V_star[b:,0]=V_star[b:,1]
    # U_star[:,0]=1
    # bottom wall
    U_star[0,:a]=0
    U_star[0,a:b]=U_star[1,a:b]
    U_star[0,b:]=0
    V_star[0,:a]=0
    V_star[0,a:b]=U_star[1,a:b]
    V_star[0,b:]=0

    # Top wall
    U_star[-1,:]=0
    V_star[-1,:]=0
    # Right wall
    U_star[:b,-1]=0
    V_star[:b,-1]=0

    U_star[b:,-1]=U_star[b:,-2]
    V_star[b:,-1]=V_star[b:,-2]
    
    
    U_star_dx=central_diff_x(U_star)
    V_star_dy=central_diff_y(V_star)
    rhs= (rho/dt*(U_star_dx+V_star_dy))
    stead=1
    const=0
    while const<=1 and  stead:
        pn = np.zeros_like(P) #[y,x]
        pn[1:-1,1:-1]=1/4 * (+P[1:-1,0:-2]+P[0:-2,1:-1]+P[1:-1,2:]+P[2:,1:-1]-dx**2*rhs[1:-1,1:-1])    
        # my inner wall (inside zeros)
        pn[:a,0]=pn[:a,1]
        pn[a:b,0]=pn[a:b,1]
        pn[b:,0]=pn[b:,1]
        # right wall 
        pn[:b,-1]=pn[:b,-2]
        pn[b:,-1]=0

        # bottom wall
        pn[0,:]=pn[1,:]
        pn[0,a:b]=0
        # left wall (initial was = 1
        # but i was told Neumman will increase convergness)
        # Top wall
        pn[-1,:]=pn[-2,:]
        P=pn
        const += 0.02
        stead=subs(pn,P)    
    pn_dx=central_diff_x(pn)
    pn_dy=central_diff_y(pn)
    U_next=( U_star - dt/rho * pn_dx)
    V_next=( V_star - dt/rho * pn_dy)
    U_next[:a,0]=0
    U_next[a:b,0]=1
    U_next[b:,0]=0
    V_next[:,0]=0

    # V_star[b:,0]=V_star[b:,1]
    # U_star[:,0]=1
    # bottom wall
    U_next[0,:a]=0
    U_next[0,a:b]=U_next[1,a:b]
    U_next[0,b:]=0
    V_next[0,:a]=0
    V_next[0,a:b]=V_next[1,a:b]
    V_next[0,b:]=0

    # Top wall
    U_next[-1,:]=0
    V_next[-1,:]=0
    # Right wall
    U_next[:b,-1]=0
    V_next[:b,-1]=0

    U_next[b:,-1]=U_next[b:,-2]
    V_next[b:,-1]=V_next[b:,-2]


    

    if t>50:
        Steady=subs(U_next,U)
    t+=1
    U=U_next
    V=V_next
    P=pn
    # Code for animation 

    # plt.contourf(X,Y,P,levels=10,vmin=0,vmax=1.6)
    # plt.quiver(X, Y, U, V)
    # plt.colorbar()
    # plt.plot(5+U[:,5],Y[:,5],color='black',linewidth=3)
    # plt.plot(20+U[:,20],Y[:,20],color='black',linewidth=3)
    # plt.plot(40+U[:,40],Y[:,40],color='black',linewidth=3)
    # plt.plot(30+U[:,30],Y[:,30],color='black',linewidth=3)
    # plt.draw()
    # plt.pause(0.05)
    # plt.clf()
print(t)
index_cut_x = int(N/10)
index_cut_y = int(M/10)
# plt.contourf(X,Y,V, alpha=0.5, cmap="plasma")
plt.contourf(X,Y,P, alpha=0.5, cmap="plasma")
# plt.contourf(X,Y,U, alpha=0.5, cmap="plasma")

plt.title(f"Contour of Velocity P and Velocity direction at iter {t}")
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



# for j in range(1,N):
    #     for k in range(1,M):
    #         U_star[j][k]=(dt/(Re*dx*dx))*(U[j+1][k]-4*U[j][k]+U[j-1][k]+U[j][k+1]+U[j][k-1])-dt*(U[j][k]*(U[j+1][k]-U[j-1][k])/(dx*2)+V[j][k]*(U[j][k+1]-U[j][k-1])/(dy*2)) + U[j][k]
    #         V_star[j][k]=(dt/(Re*dx*dx))*(V[j+1][k]-4*V[j][k]+V[j-1][k]+V[j][k+1]+V[j][k-1])-dt*(U[j][k]*(V[j+1][k]-V[j-1][k])/(dx*2)+V[j][k]*(V[j][k+1]-V[j][k-1])/(dy*2)) + V[j][k]
    
 # for i in range(1,N):
        #     for j in range(1,M):
        #         pn[i][j]=1/4*(P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]-(rho*dy)/(dt*2)*((U_star[i+1][j]-U_star[i-1][j]+V_star[i][j+1]-V_star[i][j-1])))
       
 # for j in range(1,N):
    #     for k in range(1,M):
    #         # inlet
    #         U_next[j][k]=U_star[j][k]-dt/rho *(-P[j-1][k]+P[j+1][k]/(dx*2))
    #         V_next[j][k]=V_star[j][k]-dt/rho *(-P[j][k-1]+P[j][k+1]/(dy*2))
   