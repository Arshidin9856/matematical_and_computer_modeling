# ## do i need this
# def SetTimeStep(CFL):
#     with np.errstate(divide = 'ignore'):
#         dt=CFL/np.sum([np.amax(U) / dx,np.amax(V)/ dy])
#     # Escape condition if dt is infinity due to zero velocity initially
#     if np.isinf(dt):
#         dt = CFL * (dx + dy)
#     return dt
import numpy as np
import matplotlib.pyplot as plt
import os
# Stability condition?
# cfl=0.1
T=100
dt=0.001
N=40
M=40
dx=1/N
dy=1/M
meu=0.01 #const
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
# U=np.zeros_like(X)
# V=np.zeros_like(X)
# P=np.zeros_like(X)
# U_next=np.zeros_like(X)
# V_next=np.zeros_like(X)
# U_star=np.zeros_like(X)
# V_star=np.zeros_like(X)
U=np.zeros((N+1,M+1),dtype="double")
V=np.zeros((N+1,M+1),dtype="double") 
P=np.zeros((N+1,M+1),dtype="double")
U_next=np.zeros((N+1,M+1),dtype="double")
V_next=np.zeros((N+1,M+1),dtype="double") 

U_star=np.zeros((N+1,M+1),dtype="double")
V_star=np.zeros((N+1,M+1),dtype="double") 
# def central_diff_x(f):
#     diff = np.zeros_like(f)
#     diff [1:-1,1:-1] = (f[1:-1,2:]-f[1:-1,0:-2])/(2*dx)
#     return diff
# def central_diff_y(f):
#     diff = np.zeros_like(f)
#     diff [1:-1,1:-1] = (f[2:,1:-1]-f[0:-2,1:-1])/(2*dy)
#     return diff
# def laplace(f):
#     diff = np.zeros_like(f)
#     diff [1:-1,1:-1] = (f[1:-1,0:-2]+f[0:-2,1:-1]-4*f[1:-1,1:-1]+f[1:-1,2:]+f[2:,1:-1])/(dy**2)
#     return diff


Steady=1
while Steady and t<T:
    # U_star=U_next.copy()
    # V_star=V_next.copy()
    # dt = SetTimeStep(CFL=cfl)
    print(dt,t)
    # inlet
    # for j in range(0,M+1):
    #     #inlet, outlet top
    #      #bottom wall
    #     U[j][-1]=0
    #     V[j][-1]=0
    #     P[j][-1]=P[j][-2]
    #     #top wall
    #     U[j][0]=0
    #     V[j][0]=0
    #     P[j][0]=P[j][1]
    # for i in range(0,N+1):
    #     #bottom wall
    #     U[-1][i]=1
    #     V[-1][i]=0
    #     P[-1][i]=P[-2][i]
    #     #top wall
    #     U[0][i]=0
    #     V[0][i]=0
    #     P[0][i]=P[1][i]
    

    # U_dx=central_diff_x(U)
    # U_dy=central_diff_y(U)
    # V_dx=central_diff_x(V)
    # V_dy=central_diff_y(V)
    # lap_U=laplace(U)
    # lap_V=laplace(V)
    # U_star=(U+dt*(-(U*U_dx+V*U_dy)+meu*lap_U))
    # V_star=(V+dt*(-(U*V_dx+V*V_dy)+meu*lap_V))



    for j in range(1,N):
        for k in range(1,M):
            U_star[j][k]=(dt/(Re*dx*dx))*(U[j+1][k]-4*U[j][k]+U[j-1][k]+U[j][k+1]+U[j][k-1])-dt*(U[j][k]*(U[j][k]-U[j-1][k])/(dx)+V[j][k]*(U[j][k]-U[j][k-1])/(dy)) + U[j][k]
            V_star[j][k]=(dt/(Re*dx*dx))*(V[j+1][k]-4*V[j][k]+V[j-1][k]+V[j][k+1]+V[j][k-1])-dt*(U[j][k]*(V[j][k]-V[j-1][k])/(dx)+V[j][k]*(V[j][k]-V[j][k-1])/(dy)) + V[j][k]

    # at walls
    U_star[:,0]=0
    U_star[0,:]=0
    U_star[:,-1]=0
    U_star[-1,:]=1
    V_star[:,0]=0
    V_star[0,:]=0
    V_star[:,-1]=0
    V_star[-1,:]=0
    # P[:,0]=P[:,1]
    
    # P[0,:]=P[1,:]
    # outlet 
    
    # P[:,-1]=P[:,-2]
    # outlet 
    # P[-1,:]=P[-2,:]
    # P[-1,:]=0
    
    # Solve pressure Poisson eq
    # U_star_dx=central_diff_x(U_star)
    # V_star_dy=central_diff_y(V_star)
    # rhs= ( rho/dt*(U_star_dx+V_star_dy))
    stead=1
    const=0
    while const<=1 and  stead:
        # pn[1:-1, 1:-1] = 1 / r * ((p[1:-1,2:] + p[1:-1, :-2]) / dx**2 + (p[2:, 1:-1] + p[:-2, 1:-1]) / dy**2 - D)
        pn = np.zeros_like(P)
        # pn[1:-1,1:-1]=1/4 * (+P[1:-1,0:-2]+P[0:-2,1:-1]+P[1:-1,2:]+P[2:,1:-1]-dx**2*rhs[1:-1,1:-1])    
        for i in range(1,N):
            for j in range(1,M):
                pn[i][j]=((dx*dy)/(2*dy+2*dx))*((dx**2)*(P[i][j+1]+P[i][j-1])+(dy**2)*(P[i-1][j]+P[i+1][j]))-(rho*dx*dx*dy*dy/dt)*((U_star[i+1][j]-U_star[i][j])/dx+(V_star[i][j+1]-V_star[i][j])/dy)
     
                # pn[i][j]=1/4*(+P[i][j+1]+P[i][j-1]+P[i+1][j]+P[i-1][j]-(rho*dy)/(dt*2)*((U_star[i+1][j]-U_star[i-1][j]+V_star[i][j+1]-V_star[i][j-1])))
        pn[-1,:]=0
        pn[:,-1]=pn[:,-2]
        pn[0,:]=pn[1,:]
        pn[:,0]=pn[:,1]
        # outlet 
        # pn[-1,:]=pn[-2,:]

        # at walls
        # for j in range(0,M+1):
        #     #inlet, outlet top
            
        #     p[j][0] = p[j][1]
        #     p[j][-1] = p[j][-2]
            
        # for i in range(0,N+1):
        #     #bottom wall
        #     p[0][i] = p[1][i]
        #     #top wall
        #     p[-1][i] = p[-2][i]
        # # outlet 
        P=pn
        const += 0.05
        stead=subs(pn,P)    
    for j in range(1,N):
        for k in range(1,M):
            # inlet
            U_next[j][k]=U_star[j][k]-dt/rho *(-P[j-1][k]+P[j+1][k]/(dx*2))
            V_next[j][k]=V_star[j][k]-dt/rho *(-P[j][k-1]+P[j][k+1]/(dy*2))
    # pn_dx=central_diff_x(pn)
    # pn_dy=central_diff_y(pn)
    # U_next=( U_star - dt/rho * pn_dx)
    # V_next=( V_star - dt/rho * pn_dy)

    # at walls
    U_next[:,0]=0
    V_next[:,0]=0
    # P[:,0]=P[:,1]
    
    U_next[0,:]=0
    V_next[0,:]=0
    # P[0,:]=P[1,:]
    # outlet 
    
    U_next[:,-1]=0
    V_next[:,-1]=0

    U_next[-1,:]=1
    V_next[-1,:]=0
    # P[-1,:]=P[-2,:]
    # P[-1,:]=0
    # if t>50:
    #     Steady=subs(U_next,U)
    t+=1
    U=U_next
    V=V_next
    P=pn
# print(U[0][:,0])
# print(t)
print(P)




# # print(P[-1][:][:].shape)


x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, M+1)
X,Y = np.meshgrid(x, y)


# plt.figure()
# plt.contourf(Y,X, U_next)
# plt.colorbar()
# plt.streamplot(X,Y,U,V, color='black')
# # ax.streamplot(X,Y,U[-1],V[-1], color="k")

print(f'{N,M} - N,M')
# plt.show()
index_cut_x = int(N/10)
index_cut_y = int(M/10)
X, Y = np.meshgrid(np.arange(0,N+1)/10,np.arange(0,M+1)/10)
#plt.contourf(Y,X,p, alpha=0.5, cmap="plasma")
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
