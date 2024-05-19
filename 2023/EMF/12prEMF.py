import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([0, 1],dtype=float)
y = np.array([0, 1],dtype=float)

plt.title("Lab 1 Fourier sum", loc = 'left')
plt.xlabel("x axis")
plt.ylabel("y axis")
# Fourier series 
x_axe = np.arange(0,10,0.1)
y_axe = np.arange(0,10,0.1)

l_y=[]

m=5
for i in np.arange(0,10,0.1):
    temp=[]
    for j in np.arange(0,10,0.1):
        an=bn=0
        for n in range (1,m+1):
            an+=math.pow(math.e,(math.pi*n))/(math.pow(math.e,(math.pi*n))-math.pow(math.e,(-math.pi*n)))*math.pow(math.e,(-math.pi*n*i))
            bn+=math.pow(math.e,(-math.pi*n))/(math.pow(math.e,(-math.pi*n))-math.pow(math.e,(math.pi*n)))*math.pow(math.e,(math.pi*n*i))
            res=(an+bn)*math.sin(math.pi*n*j)
        temp.append(res)
    l_y.append(temp)
y_5=np.array(l_y)  
print(l_y)  
X, Y = np.meshgrid(x_axe, y_axe)
plt.contourf(X, Y,l_y)
plt.show()
