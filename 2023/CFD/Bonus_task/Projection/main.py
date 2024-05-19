import numpy as np
import matplotlib.pyplot as plt
import os
# Stability condition?
cfl=0.8
T=100

def SetTimeStep(CFL):
    with np.errstate(divide = 'ignore'):
        dt=CFL/np.sum([np.amax(U) / dx,np.amax(V)/ dy])
    # Escape condition if dt is infinity due to zero velocity initially
    if np.isinf(dt):
        dt = CFL * (dx + dy)
    return dt
# def WriteToFile(iteration, interval):
#     p_c = P[1:-1, 1:-1]
#     u_c = U[1:-1, 1:-1]
#     v_c = V[1:-1, 1:-1]
#     if(iteration % interval == 0):
#         dir_path = os.path.join(os.getcwd(), "PUV")
#         filename = "PUV{0}.txt".format(iteration)
#         path = os.path.join(dir_path, filename)
#         with open(path, "w") as f:
#             for i in range(M-1):
#                 for j in range(N-1):
#                     f.write("{}\t{}\t{}\n".format(p_c[i,j],u_c[i,j],v_c[i,j]))

N=20
L_x=10
dx=L_x/N

M=20
L_y=10
dy=L_y/M

meu=10**-3 #const
rho=10**-1#const
Re_x=L_x*1/meu
Re_y=L_y*1/meu # isn`t 0 v_0=0`

U=np.zeros((T+1,N+1,M+1),dtype="double")
V=np.zeros((T+1,N+1,M+1),dtype="double") 
P=np.zeros((T+1,N+1,M+1),dtype="double")
U_star=np.zeros((T+1,N+1,M+1),dtype="double") # just rename U_star to F // all same
V_star=np.zeros((T+1,N+1,M+1),dtype="double") 

# firstly count starred
t=0
b=1
def subs(L_i,L_j):
    
    for i in range(N):
        for j in range(M):
            if abs(L_i[i][j]-L_j[i][j])>10**-5:
                return True     
    print('Steady')
    return False
while b:
    dt = SetTimeStep(CFL=cfl)
    print(dt,t)

    P[0][0][:]=1
    U[0][0][:]=1
    U_star[0][0][:]=1
    
    Conx=dt/Re_x
    Cony=dt/Re_y
    for i in range(T):
        for j in range(1,N):
            for k in range(1,M):
                if k>M*6/L_y and j<5*N/L_x:
                    continue
                U_star[i][j][k]=(Conx/(dx*dx)-U[i][j][k]/dx)*U[i][j+1][k]+(-Conx*2/(dx*dx)-Conx*2/(dy*dy)+U[i][j][k]/dx+V[i][j][k]/dy)*U[i][j][k]+ Conx*(U[i][j-1][k]/(dx*dx)+U[i][j][k-1]/(dy*dy))+(Conx/(dy*dy)-U[i][j][k]/dy)*U[i][j][k+1]
                V_star[i][j][k]=(Cony/(dx*dx)-U[i][j][k]/dx)*V[i][j+1][k]+(-Cony*2/(dx*dx)-Cony*2/(dy*dy)+U[i][j][k]/dx+V[i][j][k]/dy)*V[i][j][k]+ Cony*(V[i][j-1][k]/(dx*dx)+V[i][j][k-1]/(dy*dy))+(Cony/(dy*dy)-V[i][j][k]/dy)*V[i][j][k+1]
    # Solve pressure Poisson eq
    # eror=1
    iter=0
    for i in range(T):
        for j in range(1,N):
            for k in range(1,M):
                if k==0:
                    P[i][j][k+1]=P[i][j][k]
                if (k==M*6/L_y and j<=5*N/L_x):
                    P[i][j][k+1]=P[i][j][k]
                if (k==M+1 and j >= N*5/L_x):
                    P[i][j][k+1]=P[i][j][k]
                if j==N*5/L_x and k>=6*M/L_y:
                    P[i][j][k+1]=P[i][j][k]
                if k>M*6/L_y and j<5*N/L_x:
                    continue
                P[i+1][j][k]=((dx*dy)/(2*dy+2*dx))*((dx**2)*(P[i][j][k+1]+P[i][j][k-1])+(dy**2)*(P[i][j-1][k]+P[i][j+1][k]))-(rho*dx*dx*dy*dy/dt)*((U_star[i][j+1][k]-U_star[i][j][k])/dx+(V_star[i][j][k+1]-V_star[i][j][k])/dy)
        # iter += 1        
        # b=subs(P[iter],P[iter-1])
        # eror=max(abs(P[iter]-P[iter-1]))
    for i in range(T):
        for j in range(1,N):
            for k in range(1,M):
                if k>M*6/L_y and j<5*N/L_x:
                    continue 
                U[i+1][j][k]=U_star[i][j][k]+dt*(P[i][j+1][k]-P[i][j][k])/(rho*dx)
                V[i+1][j][k]=V_star[i][j][k]+dt*(P[i][j][k+1]-P[i][j][k])/(rho*dy)
    
    # WriteToFile( t, 5)
    if t%5==0 :
        x = np.linspace(0, L_x, N)
        y = np.linspace(0, L_y, M)
        X,Y = np.meshgrid(x, y)

        # #### Determine indexing for stream plot (10 points only)
        index_cut_x = int(N/10)
        index_cut_y = int(M/10)

        # #### Create blank figure
        fig = plt.figure(figsize=(16, 8))
        ax = plt.axes(xlim=(0,N), ylim=(0, M))

        # #### Create initial contour and stream plot as well as color bar
        # p_p, u_p, v_p = read_datafile(0)
        ax.set_xlim([0, L_x])
        ax.set_ylim([0, L_y])
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$y$", fontsize=12)
        ax.set_title("Frame No: 0")

        cont=ax.contourf(X,Y, P[-1][0:N,0:M])

        ax.streamplot(X[::index_cut_x, ::index_cut_y],\
                            Y[::index_cut_x, ::index_cut_y],\
                            U[-1][:N:index_cut_x, :M:index_cut_y],\
                            V[-1][:N:index_cut_x, :M:index_cut_y],\
                            color="k")
        fig.colorbar(cont)
        fig.tight_layout()
        plt.show()
    t+=1
    b=subs(U[t],U[t-1])
# print(U)
# print(V)





# print(P[-1][:][:].shape)


x = np.linspace(0, L_x, N)
y = np.linspace(0, L_y, M)
X,Y = np.meshgrid(x, y)

# #### Determine indexing for stream plot (10 points only)
index_cut_x = int(N/10)
index_cut_y = int(M/10)

# #### Create blank figure
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(xlim=(0,N), ylim=(0, M))

# #### Create initial contour and stream plot as well as color bar
# p_p, u_p, v_p = read_datafile(0)
ax.set_xlim([0, L_x])
ax.set_ylim([0, L_y])
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)
ax.set_title("Frame No: 0")

cont=ax.contourf(X,Y, P[-1][0:N,0:M])

ax.streamplot(X[::index_cut_x, ::index_cut_y],\
                       Y[::index_cut_x, ::index_cut_y],\
                       U[-1][:N:index_cut_x, :M:index_cut_y],\
                       V[-1][:N:index_cut_x, :M:index_cut_y],\
                       color="k")
fig.colorbar(cont)
fig.tight_layout()
plt.show()