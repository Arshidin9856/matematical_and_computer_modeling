import numpy as np
import matplotlib.pyplot as plt
import os
# Stability condition?
cfl=0.1
T=100
dt=0.001

## do i need this
# def SetTimeStep(CFL):
#     with np.errstate(divide = 'ignore'):
#         dt=CFL/np.sum([np.amax(U) / dx,np.amax(V)/ dy])
#     # Escape condition if dt is infinity due to zero velocity initially
#     if np.isinf(dt):
#         dt = CFL * (dx + dy)
#     return dt


dx=0.1
dy=0.1
N=int(1/dx)

M=int(1/dy)

meu=0.1 #const
rho=1#const
Re_x=10
Re_y=10
U=np.zeros((N+1,M+1),dtype="double")
V=np.zeros((N+1,M+1),dtype="double") 
P=np.zeros((N+1,M+1),dtype="double")
U_next=np.zeros((N+1,M+1),dtype="double")
V_next=np.zeros((N+1,M+1),dtype="double") 

U_star=np.zeros((N+1,M+1),dtype="double")
V_star=np.zeros((N+1,M+1),dtype="double") 

# firstly count starred
t=0
# b=1
def poisson(p,us,vs,k,m,dt_p,const=0):
 pn = np.empty_like(p)
 stead=1
 while stead:
    pn = p.copy()
    
    for i in range(1,k):
        for j in range(1,m):
           p[i][j]=((dx*dy)/(2*dy+2*dx))*((dx**2)*(pn[i][j+1]+pn[i][j-1])+(dy**2)*(pn[i-1][j]+pn[i+1][j]))-(rho*dx*dx*dy*dy/dt)*((us[i+1][j]-us[i][j])/dx+(vs[i][j+1]-vs[i][j])/dy)
    # outlet 
    p[0,a:]=1
    p[-1,a:]=0
    p[-1,:b]=0
    # at walls
    p[0,:a]=p[1,:a] # left wall
    
    p[:,0]=p[:,1] # upper
    p[:,-1]=p[:,-2] #bottom

    p[-1,b:a]=p[-2,b:a] #right wall 
    const += dt_p
    stead=subs(pn,p)    
 return p
def subs(L_i,L_j):
    
    for i in range(N):
        for j in range(M):
            if abs(L_i[i][j]-L_j[i][j])>10**-5:
                return True     
    print('Steady')
    return False
Steady=1
while Steady:
    U_star=U_next.copy()
    V_star=V_next.copy()
    U=U_next.copy()
    V=V_next.copy()
    # dt = SetTimeStep(CFL=cfl)
    print(dt,t)
    a= int(0.7/dy)
    b=int(0.3/dy)
    Conx=dt/Re_x
    Cony=dt/Re_y
    # inlet
    U[0,a:]=1
    V[0,a:]=0
    P[0,a:]=1
    # outlet 
    U[-1,a:]=U[-2,a:]
    V[-1,a:]=0
    P[-1,a:]=0
    
    U[-1,:b]=U[-2,:b]
    V[-1,:b]=0
    P[-1,:b]=0
    # at walls
    U[0,:a]=0
    V[0,:a]=0
    P[0,:a]=P[1,:a]
    
    U[:,0]=0
    V[:,0]=0
    P[:,0]=P[:,1]
    
    U[:,-1]=0
    V[:,-1]=0
    P[:,-1]=P[:,-2]

    U[-1,b:a]=0
    V[-1,b:a]=0
    P[-1,b:a]=P[-2,b:a]
    # outlet 
    U_next=U.copy()
    V_next=V.copy()
    for j in range(1,N):
        for k in range(1,M):
            U_star[j][k]=(dt/(Re_x*dx*dx))*(U[j+1][k]-2*U[j][k]+U[j-1][k])+(dt/(Re_x*dy*dy))*(U[j][k+1]-2*U[j][k]+U[j][k-1])-dt*U[j][k]*(U[j+1][k]-U[j][k])/dx-dt*V[j][k]*(U[j][k+1]-U[j][k])/dy + U[j][k]
            V_star[j][k]=(dt/(Re_x*dx*dx))*(V[j+1][k]-2*V[j][k]+V[j-1][k])+(dt/(Re_x*dy*dy))*(V[j][k+1]-2*V[j][k]+V[j][k-1])-dt*U[j][k]*(V[j+1][k]-V[j][k])/dx-dt*V[j][k]*(V[j][k+1]-V[j][k])/dy+V[j][k]
    
    # Solve pressure Poisson eq
    eror=1
    iter=0
    
    pn= poisson(P,U_star,V_star,N,M,0.02)
    for j in range(1,N):
        for k in range(1,M):
            # inlet
            U_next[j][k]=U_star[j][k]+dt*(pn[j-1][k]-pn[j][k])/(rho*dx)
            V_next[j][k]=V_star[j][k]+dt*(pn[j][k-1]-pn[j][k])/(rho*dy)
    t+=1
    Steady=subs(U_next,U)
# print(U[0][:,0])
# print(t)





# # print(P[-1][:][:].shape)


x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, M+1)
X,Y = np.meshgrid(x, y)


# plt.figure()
# plt.contourf(Y,X, U_next)
# plt.colorbar()
# plt.streamplot(X,Y,U,V, color='black')
# # ax.streamplot(X,Y,U[-1],V[-1], color="k")

# plt.show()
# X, Y = np.meshgrid(np.arange(0,N+1)/10,np.arange(0,M+1)/10)
#plt.contourf(Y,X,p, alpha=0.5, cmap="plasma")
plt.contourf(Y,X,V_next, alpha=0.5, cmap="plasma")
plt.title("Contour of Velocity U and Velocity direction")
plt.colorbar()
plt.quiver(X[::1, ::1], Y[::1, ::1], U[::1, ::1], V[::1, ::1])
#plt.streamplot(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#print(un)
#print(us)
#print(vs)