import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([0, 1],dtype=float)
y = np.array([0, 1],dtype=float)
z = np.array([0, 1],dtype=float)


plt.title("Lab 1 Fourier sum", loc = 'left')
plt.xlabel("x axis")
plt.ylabel("y axis")
# Fourier series 
x_axe = np.arange(0,1,0.1)
y_axe = np.arange(0,1,0.1)
z_axe = np.arange(0,1,0.1)

U=np.zeros((10,10,10))

quant=15
for i in range(0,10):
    # matrix_of_x=[]
    for j in range(0,10):
        # row_of_z_for_y=[]
        for k in range(0,10):
            an=bn=0
            res=0
            for n in range (1,quant+1):
                for m in range (1,quant+1):
                    pie=math.pi
                    root=math.sqrt((pie*n)**2 + (pie*m)**2)
                    an+=math.pow(math.e,-root)* 4 * ((-1)**(m+1)+1) * ((-1)**(n+1)+1) * math.pow(math.e,root*i*0.1) / (pie**2 * n*m * (math.pow(math.e,-root)-math.pow(math.e,root)))
                    bn+=math.pow(math.e,root)* 4 * ((-1)**(m+1)+1) * ((-1)**(n+1)+1) * math.pow(math.e,-root*i*0.1) / (pie**2 * n*m * (math.pow(math.e,root)-math.pow(math.e,-root)))
                    res+=(an+bn)*math.sin(pie*n*j*0.1)*math.sin(pie*m*k*0.1)

            # row_of_z_for_y.append(res)
            U[i][j][k]=res
    # matrix_of_x.append(row_of_z_for_y)
    # U.append(matrix_of_x)
# Result=np.array(U)  
  
X, Y,Z = np.meshgrid(x_axe, y_axe,z_axe)
# plt.contourf(Y, Z,Result[5])
plt.contourf(U[5][:][:])
plt.show()
