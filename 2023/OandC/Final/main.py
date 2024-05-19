import math
import random
import numpy as np
import time
# Check if our point is satisfies
# Need to check before making step in direction * t 

def Feasible(P):
    if P[0]<=1: return False
    return  -math.log(P[0]-1)-P[1]<-1 and 2*P[0]+P[1]<6        

#Second derivative approximation
# Return matrix of second derivatives( Hessian operator) n*n
def sec_derivatives(P,h=1e-6):
    res=np.zeros((n,n))
    z=P.copy()
    y=P.copy()
    term=find_valie(y)
    for col_ind in range(n):
        y[col_ind]+=h    
        term1=find_valie(y)
        y[col_ind]-=2*h    
        term2=find_valie(y)
        for row_ind in range(n):
            if row_ind==col_ind:
                res[col_ind,row_ind]=(term1-2*term+term2)/(h**2)
            else:
                z[col_ind]+=h
                z[row_ind]-=h
                term3=find_valie(z)
                z[col_ind]-=h    
                term4=find_valie(z)
                res[col_ind,row_ind]=(term1-term3-term+term4)/h**2
        z=P.copy()
        y=P.copy()
    return res     
# Return vector of first der in form n*1
def first_derivatives(P,h=1e-6):
    res=np.zeros((n,1))
    z1=P.copy()
    for col_ind in range(n):
        z1[col_ind]+=h    
        term1=find_valie(z1)
        z1[col_ind]-=2*h    
        term2=find_valie(z1)
        res[col_ind][0]=(term1-term2)/( 2*h)
        z1=P.copy()
    return res     
# Our function with logarithm barierr method
def find_valie(x):
    return x[0]+x[1]*2- 1/Qual * (math.log(math.log(x[0]-1)+x[1]-1)+math.log(-2*x[0]-x[1]+6))           
# Returns direction of most decrease and decrement Y
# Y used to stop criteria
def Newton_step(x):
        # 2grad = [n*n], grad [n*1] ==>  dx = [n*1] 
        # direction = - sec_der^-1 * first_der 
        try:
            inverse = np.linalg.inv(sec_derivatives(x)) 
        except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
            print("NOT INVERTIBLE !!!")
            exit()
        else:
            v=np.zeros((n,1),dtype=float)
            df=first_derivatives(x)
            v=np.matmul(-inverse,df)
            decrement=np.matmul(df.T[0],np.matmul(inverse,df))
            # I transpose v (n*1 --> 1*n)for computational purposes(because my point is 1*n and i need to sum them)
        return v.T[0],decrement
def backtraking_search(point):
    t=1
    #  Needs to not overcome our boundary region. So we decrease step until its Feasible
    while not Feasible(point+t*direction): t=beta*t 
    # Then decrease t until i find less value 
    future_value=find_valie((point+t*direction))
    line=(find_valie(point)+alpha*t*(np.matmul(direction,first_derivatives(point))))[0]
    while True:
        if future_value<line:
            break
        t=beta*t
        future_value=find_valie((point+t*direction))
        line=(find_valie(point)+alpha*t*(np.matmul(direction,first_derivatives(point))))[0]
        if t==0:
            print('didnt found step t')
            exit()
    return t
# My results in diff points
# array([2.79141856, 0.41700978])
# array([2.79146765, 0.41697165])
# array([2.7914669 , 0.41697204])
# This points satisfy conditions

point=np.array([1.3,2.8]) # we choose only Feasibles point x>1 and 1<y<2
# Very hard to find Feasible points
if not Feasible(point):
    print ("Notfeasible")
    exit()
Const_num=2     # Number of constraines
history=[]      # All my values
Points=[point]  # All Points
n=2             # number of variables
alpha=0.1         
beta=0.5
epsilon=1e-5
maxiter=500

Qual=0.01          # Quality of our logarithm boundary
Convergence=0
while Const_num/Qual>epsilon:
    Y=1
    iter=0
    print(Points[-1])
    while  Y**2/2>epsilon and iter<maxiter:
        direction,Y=Newton_step(Points[-1])
        T=backtraking_search(Points[-1])
        
        Points.append(Points[-1]+T*direction)
        history.append(find_valie(Points[-1]))
        iter+=1
    Qual*=2
    Convergence+=1

print(f"my points {Points[-1]}")
print(f"my values {history[-1]}")
print(f"Converge in n iter {Convergence}")

import matplotlib.pyplot as plt
from scipy.optimize import fsolve
x_values = np.linspace(0, 3, 100)
y_values = np.linspace(0, 3, 100)
X, Y = np.meshgrid(x_values, y_values)
def f(x,y):
    return x+2*y
def constraint1(x,y):
 
    return -np.log(x-1)-y+1

def constraint2(x,y):
    return 2*x+y-6
# Plot the functions
# Plot points, steps
for x, fx in zip(Points, history):
    plt.plot(x[0], x[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
plt.contour(X, Y, f(X, Y), levels=20, cmap='viridis')
plt.contour(X, Y, constraint1(X, Y), levels=[0], colors='red')
plt.contour(X, Y, constraint2(X, Y), levels=[0], colors='green')

# Find the intersection point
intersection_point = fsolve(lambda z: [constraint1(z[0], z[1]), constraint2(z[0], z[1])], x0=[1.5, 0.5])

# Plot the intersection point
plt.scatter(intersection_point[0], intersection_point[1], color='blue', label='Intersection Point')
plt.text(1.5, 0.5, 'x + y = 0', color='red', fontsize=8)
plt.text(1.5, 1.5, '2 - x - y = 0', color='green', fontsize=8)

# Set plot labels and legend
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.grid(True)
plt.show()