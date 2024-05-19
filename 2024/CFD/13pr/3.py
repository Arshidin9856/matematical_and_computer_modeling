import numpy as np
import matplotlib.pyplot as plt
dt = 0.001
dx = 0.1
dy = 0.1
Re = 10
rho = 1
const = 0
k = int(1/dx)
m = int(1/dy)
u = np.zeros((k+1, m+1)) #(31,31)
v = np.zeros((k+1, m+1))
un = np.zeros((k+1, m+1)) #(31,31)
vn = np.zeros((k+1, m+1))
p = np.zeros((k+1, m+1))
us = np.zeros((k+1, m+1))
vs = np.zeros((k+1, m+1))
def poisson(p,us,vs,k,m,dt_p,const=0):
 pn = np.empty_like(p)
 while const<=1:
    pn = p.copy()
    
    for i in range(1,k):
        for j in range(1,m):
           p[i][j]=((dx*dy)/(2*dy+2*dx))*((dx**2)*(pn[i][j+1]+pn[i][j-1])+(dy**2)*(pn[i-1][j]+pn[i+1][j]))-(rho*dx*dx*dy*dy/dt)*((us[i+1][j]-us[i][j])/dx+(vs[i][j+1]-vs[i][j])/dy)
        #p[i][j] = ((dy*dy*(pn[i+1][j] + pn[i-1][j])) + (dx*dx*(pn[i
        #    p[i][j]=0.25*(pn[i-1][j]+pn[i+1][j]+pn[i][j-1]+pn[i][j+1]-(
    for j in range(0,m+1):
        #inlet, outlet top
        if j >= 7:
            p[0][j] = 1
            p[k][j] = 0
        #left wall
        else:
            p[0][j] = p[1][j]
            if j > 3: #right wall
               p[k][j] = p[k-1][j] 
            else: #outlet bot
                p[k][j] = 0
    
    for i in range(0,k+1):
        #bottom wall
        p[i][0] = p[i][1]
        #top wall
        if i > 0:
           p[i][m] = p[i][m-1]
        #print(const) 
    const += dt_p
        
 return p
while const<=1:
    us = un.copy()
    vs = vn.copy()
    u = un.copy()
    v = vn.copy() 
    #walls, inlet
    for j in range(0,m+1):
    #inlet, outlet top
        if j >= 7:
            u[0][j] = 1
            v[0][j] = 0
            p[0][j] = 1
            u[k][j] = u[k - 1][j]
            v[k][j] = 0
            p[k][j] = 0
            
 #left wall
        else:
            u[0][j] = 0
            v[0][j] = 0
            p[0][j] = p[1][j]
            if j > 3: #right wall
                u[k][j] = 0
                v[k][j] = 0
                p[k][j] = p[k-1][j] 
            else: #outlet bot
                u[k][j] = u[k - 1][j]
                v[k][j] = 0
                p[k][j] = 0
        
    for i in range(0,k+1):
 #bottom wall
        u[i][0] = 0
        v[i][0] = 0
        p[i][0] = p[i][1]
        #top wall
        if i > 0:
            u[i][m] = 0
            v[i][m] = 0
            p[i][m] = p[i][m-1]
 
 
 #print(u)
 #print(p)
    un = u.copy()
    #print(un)
    for i in range(1,k):
       for j in range(1,m):
    #u-v star
        #print(us) 
        # us[i][j] = u[i][j] + (dt/Re) * (((u[i+1][j] - 2*u[i][j] + u[i-1]
        # vs[i][j] = v[i][j] + (dt/Re) * (((v[i+1][j] - 2*v[i][j] + v[i-1]
        us[i][j]=(dt/(Re*dx*dx))*(u[i+1][j]-2*u[i][j]+u[i-1][j])+(dt/(Re*dy*dy))*(u[i][j+1]-2*u[i][j]+u[i][j-1])-dt*u[i][j]*(u[i+1][j]-u[i][j])/dx-dt*v[i][j]*(u[i][j+1]-u[i][j])/dy + u[i][j]
        vs[i][j]=(dt/(Re*dx*dx))*(v[i+1][j]-2*v[i][j]+v[i-1][j])+(dt/(Re*dy*dy))*(v[i][j+1]-2*v[i][j]+v[i][j-1])-dt*u[i][j]*(v[i+1][j]-v[i][j])/dx-dt*v[i][j]*(v[i][j+1]-v[i][j])/dy+v[i][j]
          
    pn = poisson(p,us,vs,k,m,0.02)
    #print(pn)
    for i in range(1,k):
     for j in range(1,m):
    
        un[i][j] = (-dt/rho) * ((pn[i][j] - pn[i-1][j])/dx)+ us[i][j]
        vn[i][j] = (-dt/rho) * ((pn[i][j] - pn[i][j-1])/dy)+ vs[i][j]
        #print("us=",us[i][j], i ,j)
 #print("pn=",pn[i][j], i ,j)
 #print("un=",un[i][j], i ,j)
 #print(const)
    const += dt
 
#X, Y = np.meshgrid(np.linspace(0,1,k+1),np.linspace(0,1,m+1)) 
X, Y = np.meshgrid(np.arange(0,k+1)/10,np.arange(0,m+1)/10) 
#plt.contourf(Y,X,p, alpha=0.5, cmap="plasma") 
plt.contourf(Y,X,vn, alpha=0.5, cmap="plasma") 
plt.title("Contour of Velocity U and Velocity direction")
plt.colorbar()
plt.quiver(X[::1, ::1], Y[::1, ::1], u[::1, ::1], v[::1, ::1]) 
#plt.streamplot(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
#print(un) 
#print(us)
#print(vs
