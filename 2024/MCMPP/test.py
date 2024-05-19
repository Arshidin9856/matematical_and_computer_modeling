
import numpy as np
# A=np.full((3,4),5)
# B=[1,1,1]
# A.T[1]=B
# A[2]=8
# np.diag(np.ones(), 1) 
n=5
dx=0.01
dy=0.01
dz=0.01
count=1
B=np.eye(n)*(2/dx**2+2/dy**2 + 2*np.cos(np.pi*count/n)/dz**2)
C=np.diag(np.ones(n-1)*(-1/dy**2), 1) + np.diag(np.ones(n-1)*(-1/dy**2), -1) 
print(B.shape)
print(C.shape)
n=11
if n%3==0:
    raise IndexError
# boundary
a_x=n//3
b_x=round(n/3)+n//3+1
print(a_x,b_x)
print(n-b_x,a_x)
for count in range(1,n):
# First step   find a_n then
        if count >a_x and count<=b_x:
            print(count)
# print(np.diag(np.array([1,2,3,4,5]), -1) )    
# betha1=[[0]*n,[0]*n] # column
# alpha1=[[[0]*n]*n,[[0]*n]*n] # matrix
# # A=np.eye(n)/dx**2
# # C=np.eye(n)/dx**2
# # D=np.zeros(n)
# print([0]*7+[500]*3)
# B=np.eye(n)*(2/dx**2+2/dy**2 + 2*np.cos(np.pi*count/n)/dz**2) 

# alpha_temp=np.dot(np.linalg.inv(-(B+np.dot(C,alpha1[-1]))),A)
# bettha_temp=np.dot(np.linalg.inv(-(B+np.dot(C,alpha1[-1]))),D+betha1[-1])

# alpha1.append(alpha_temp.tolist())
# betha1.append(bettha_temp)
# print(alpha1)