import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


t=100
dt=1/t
n=10
dx=2/n
l=np.zeros((t+1,n+1))
y=[]
for i in range(t+1):
    y.append(i*dt)
x=[]
for i in range(n+1):
    x.append(i*dx)
b=False
l[0][0]=1
l[0][n]=0
for i in range(1,t):
    l[i][0]=1
    for j in range(1,n):
        if i ==0:
            l[0][j]=0
        l[i+1][j]=l[i][j+1]*((dt)/dx)-l[i][j-1]*((dt)/dx)+l[i-1][j]
#         if not b and abs(l[i][j]-l[i-1][j])<10**-4 and str(l[i][j])[len(str(l[i][j]))-4]=='e'and l[i][j]!=0 and str(l[i+1][j]).split('.')[0]==str(l[i][j]).split('.')[0] and i>50:
#             # print(l)
#             print("\nHELLOOOOO",l[i+1][j],l[i][j],(i,j))
#             x_mesh, t_mesh = np.meshgrid(x, y)

# # Calculate f(x, t) for each combination ofx and t
#             result = np.copy(l)

# # Create a 3D plot
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')

#             # Plot the function over the meshgrid
#             ax.plot_surface(x_mesh, t_mesh, result, cmap='viridis')

#             # Set labels
#             ax.set_xlabel('X-axis')
#             ax.set_ylabel('Time (t)')
#             ax.set_zlabel('f(x, t)')

#             # Show the plot
#             plt.show()
#             b=True
#             break           
            
    l[i][n]=0
if not b:
    print(l)   
    x_mesh, t_mesh = np.meshgrid(x,y)

    # Calculate f(x, t) for each combination of x and t
    

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the function over the meshgrid
    ax.plot_surface(x_mesh, t_mesh, l, cmap='viridis')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('f(x, t)')

    # Show the plot
    plt.show()