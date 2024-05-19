from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
#Burgers eequation using fractional step method
n=26
if n%3==0:
    raise IndexError
# boundary
a_x=n//3
b_x=round(n/3)+n//3+1
print(a_x,b_x)
# coeff
T=1000
dx=1/(n-1)
dy=1/(n-1)
dt=1/(n-1)
RE=5
eps=10**-5
#matrices
U_prev=np.zeros((n,n))
V_prev=np.zeros((n,n))
U_temp=np.zeros((n,n))
V_temp=np.zeros((n,n))
U_n=np.zeros((n,n))
V_n=np.zeros((n,n))

X, Y = np.meshgrid(np.arange(0,n),np.arange(0,n))

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

C=-1/(2*((RE*dx)**2))
D_U=np.zeros((n,n))
D_V=np.zeros((n,n))

A_U=np.zeros((n,n))
A_V=np.zeros((n,n))
B_U=np.zeros((n,n))
B_V=np.zeros((n,n))

# def get_ABD(f,V):
#                        #y     x
#     d=np.zeros_like(f) # f[1:-1,1:-1]
#     a=f/(2*dx)-1/(2*((RE*dx)**2))
#     b=1/dt-f/(2*dx)+1/((RE*dx)**2)
#     d[1:-1,1:-1]=-f[1:-1,1:-1]/dt + (-1/(2*(RE**2)) - 1/(RE*dy)**2) *(f[2:,1:-1]-2*f[1:-1,1:-1]+f[0:-2,1:-1]) + 3*V[1:-1,1:-1] *(f[2:,1:-1]-f[1:-1,1:-1])/(2*dy) 
#     return a,b,d
# def getD_second(f,f_prev,V):
#                        #y     x
#     d=np.zeros_like(f) # f[1:-1,1:-1]
#     d[1:-1,1:-1]=1/RE**2 * (f_prev[2:,1:-1]- 2* f_prev[1:-1,1:-1] + f_prev[0:-2,1:-1]) - V[1:-1,1:-1] *  (f_prev[2:,1:-1]-f_prev[1:-1,1:-1]) - f[1:-1,1:-1]/dt         
#     return d
iter_thomas=0
max_d=10
while   max_d> eps and iter_thomas<100:
# First step for U 
    # A_U,B_U,D_U=get_ABD(U_prev,V_prev) # For 1 _UV step AB for 1_U step D
    # A_V,B_V,D_V=get_ABD(V_prev,V_prev) # for 2_UV step for 1_V step D
    for i in range(n):
        for j in range(n):
    
            A_U[i,j]=U_prev[i,j]/(2*dx)-1/(2*((RE*dx)**2))
            B_U[i,j]=1/dt-U_prev[i,j]/(2*dx)+1/((RE*dx)**2)
    for i in range(n):
        for j in range(1,n-1):
            D_U[i,j]=-U_prev[i,j]/dt + (-1/(2*(RE**2)) - 1/(RE*dy)**2) *(U_prev[i,j+1]-2*U_prev[i,j]+U_prev[i,j-1]) + 3*V_prev[i,j] *(U_prev[i,j+1]-U_prev[i,j])/(2*dy) 
    for i in range(n):
        for j in range(n):
            A_V[i,j]=V_prev[i,j]/(2*dx)-1/(2*((RE*dx)**2))
            B_V[i,j]=1/dt-V_prev[i,j]/(2*dx)+1/((RE*dx)**2)
    for i in range(n):
        for j in range(1,n-1):
            D_V[i,j]=-V_prev[i,j]/dt + (-1/(2*(RE**2)) - 1/(RE*dy)**2) *(V_prev[i,j+1]-2*V_prev[i,j]+V_prev[i,j-1]) + 3*V_prev[i,j] *(V_prev[i,j+1]-V_prev[i,j])/(2*dy) 
    # D_V=-V_prev/dt + (-1/(2*(RE**2)) - 1/(RE*dy)**2) *(laplace(V_prev)) + 3*V_prev *(central_diff_y(V_prev))/(2*dy) 
    
    
# For U 1 step
    for i in range(1,n-1):
        if i >a_x and i<=b_x:
            bethau=[0,1]
        else:
            bethau=[0,0]

        p_nu=[0,0]
        p_nv=[0,0]
        bethav=[0,0]
        alphau=[0,0]
        alphav=[0,0]
        # if  j>a_x and j<=b_x:
        #     alphau=[0,0]
        #     alphav=[0,0]
        # else:
        #     alphau=[0,0]
        #     alphav=[0,0]
        # bethau=[0,-1]
        # bethav=[0,0]
        # p_nu=[0,0]
        # p_nv=[0,0]

        for j in range(1,n):
            alphau.append(-A_U[i][j]/(B_U[i][j]+C*alphau[-1]))
            bethau.append((D_U[i][j]-C*bethau[-1])/(B_U[i][j]+C*alphau[-1]))
            alphav.append(-A_U[i][j]/(B_U[i][j]+C*alphav[-1]))
            bethav.append((D_V[i][j]-C*bethav[-1])/(B_U[i][j]+C*alphav[-1]))
        if i>b_x:
            p_nu=[0,bethau[-1]/(1-alphau[-1])]
            p_nv=[0,bethav[-1]/(1-alphav[-1])]   
            
        
        for j in range(n-1,0,-1):
            p_nu.append(alphau[j]*p_nu[-1]+bethau[j])
            p_nv.append(alphav[j]*p_nv[-1]+bethav[j])
        
        tempu=np.array(p_nu[1:])
        tempv=np.array(p_nv[1:])
        
        U_temp.T[i]=tempu[::-1]
        V_temp.T[i]=tempv[::-1]  
    
    
## second step
    # D_U=getD_second(U_temp,U_prev,V_prev)
    # D_V=getD_second(V_temp,V_prev,V_prev)
    for i in range(n):
        for j in range(1,n-1):
            D_U[i,j]=1/((RE*dy)**2) * (U_prev[i,j+1]- 2* U_prev[i,j] + U_prev[i,j-1]) - V_prev[i,j] *  (U_prev[i,j+1]-U_prev[i,j]) - U_temp[i,j]/dt   
    for i in range(n):
        for j in range(1,n-1):
            D_V[i,j]=1/((RE*dy)**2) * (V_prev[i,j+1]- 2* V_prev[i,j] + V_prev[i,j-1]) - V_prev[i,j] *  (V_prev[i,j+1]-V_prev[i,j]) - V_temp[i,j]/dt 
    
    # D 2 step U
    for j in range(1,n-1):
        if  j>a_x and j<=b_x:
            alphau=[0,-1]
            alphav=[0,-1]
        else:
            alphau=[0,0]
            alphav=[0,0]
        bethau=[0,0]
        bethav=[0,0]
        p_nu=[0,0]
        p_nv=[0,0]
                

        for i in range(1,n):
            alphau.append(-A_V[i][j]/(B_V[i][j]+C*alphau[-1]))
            bethau.append((D_U[i][j]-C*bethau[-1])/(B_V[i][j]+C*alphau[-1]))
            
            alphav.append(-A_V[i][j]/(B_V[i][j]+C*alphav[-1]))
            bethav.append((D_V[i][j]-C*bethav[-1])/(B_V[i][j]+C*alphav[-1]))
        for i in range(n-1,0,-1):
            p_nu.append(alphau[i]*p_nu[-1]+bethau[i])
            p_nv.append(alphav[i]*p_nv[-1]+bethav[i])

        tempu=np.array(p_nu[1:])
        tempv=np.array(p_nv[1:])

        U_n[j]=tempu[::-1]
        V_n[j]=tempv[::-1] 
    


# exit condition
    max_d=0
    for i in range(n):
        for j in range(n):
            if max_d<abs(U_n[i][j]-U_prev[i][j]):
                max_d=abs(U_n[i][j]-U_prev[i][j])
    # if max>0.008:
    if iter_thomas in [0,15,35,70]:
        plt.quiver(X, Y, U_n, V_n)
        
        plt.title(f"Contour of Velocity U and Velocity direction at iter {iter_thomas}")
        plt.show()
    # plt.pause(0.001)
    U_prev=np.copy(U_n)
    V_prev=np.copy(V_n)
    iter_thomas+=1
    # print(iter_thomas, max_d)
print(U_n)
#plt.contourf(Y,X,p, alpha=0.5, cmap="plasma")
plt.title("Contour of Velocity U and Velocity direction")

plt.quiver(X, Y, U_n, V_n)
# plt.streamplot(X, Y, U_n, V_n, color='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# def graph(T, n=n):
#     x = np.linspace(0, 1, n)
#     y = np.linspace(0, 1, n)
#     z = np.linspace(0, 1, n)
#     x, y, z = np.meshgrid(x, y, z, indexing='ij')

#     fig = go.Figure(data=[go.Volume(
#         x=x.flatten(),
#         y=y.flatten(),
#         z=z.flatten(),
#         value=T.flatten(),
#         isomin=T.min(),
#         isomax=T.max(),
#         opacity=0.1,
#         surface_count=21,
#         colorscale='Viridis')])
#     fig.update_layout(scene=dict(
#         xaxis=dict(title='X'),
#         yaxis=dict(title='Y'),
#         zaxis=dict(title='Z')))
#     fig.show()
# graph(U_n)

