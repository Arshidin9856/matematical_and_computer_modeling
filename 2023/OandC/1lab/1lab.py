import numpy as np
import random


# random input

n=100
m=100
max_f=np.zeros(n)
for i in range(n):
    max_f[i]=-random.randint(0,100)
Conditions= np.zeros((m,n))
for j in range (m):
    
    for i in range(n):
        Conditions[j][i]=random.randint(0,100)  
b = np.zeros(m+1,float)

for j in range(m):
    b[j]=random.randint(0,10)

# input by hand
# exaple 1
# n=2
# m=2
# Conditions = np.array([[ 1,1], 
#                        [ 3,2] ])
# max_f = np.array([ -1,-2])
# b = np.array([0,6,12],float)

# example 2
# n=2
# m=3

# Conditions = np.array([[3, 2], 
#                        [3, 5], 
#                        [5,6]])
# b = np.array([0,600, 800,1100],float)
# max_f = np.array([ -30,-40])


# example 3

n=3
m=3

Conditions = np.array([[10, 2,1], 
                       [7, 3,2], 
                       [2,4,1]])
b = np.array([0,100, 72,80],float)
max_f = np.array([ -22,-6,-2])


sol=np.zeros(m)
for i in range(m):
    sol[i]=i+n

# make table from coefficients to work with(without column b)
def make_table(A,C,B):
    matrix=np.zeros([m+1,m+n])

    matrix[0]=np.concatenate((C,np.zeros(m)))
    for i in range (m):
        S=np.zeros(m)
        S[i]=1
        matrix[i+1]=np.concatenate((A[i],S))
    #     for j in range(m):
    return matrix
# if there's negative coef in objective function
def check_neg(A):
   
    for i in range(n+m):
        if A[0][i]<0: return True
    return False    
# choose pivot (col)
def max_neg(A):
    min=A[0][0]
    Ind=0
    for i in range(n+m):
        if A[0][i]<min: 
            min=A[0][i]
            Ind=i
    return Ind   
# choose pivot (row)
def min_quot(A,ind):
    min=b[1]/A[1][ind]
    j=1
    for i in range(1,m+1):
        if b[i]/A[i][ind]<min and b[i]/A[i][ind]>=0: # > or >=0
            min=b[i]/A[i][ind]
            j=i
    return  j      
# add index of pivot to solution  
def change_sol(row,col):
    sol[row-1]=col
# make 1 in pivot coeff
def div(vector,q):
    for i in range(n+m):
        vector[i]=vector[i]/q
    return vector
# add a to b (row), return new row b
def add(a,b):
    for i in range(len(a)):
        b[i]+=a[i]
    return b
# function for making zerros in all rows in pivot column
def sumR(B,row,col):
                # i[a] row to make 1
        print(b,row,col)
        # print(B)

        for i in range(m+1):

            
            if i==row:
                
                b[i]=b[i]/B[row][col]
                B[i]=div(B[i],B[row][col])
                
            
            else:
                quot=-B[i][col]/B[row][col]
                new=B[row].copy()

                for j in range(n+m):
                    new[j]=B[row][j]*quot
                
                B[i]=add(new,B[i])
                b[i]=b[row]*quot+b[i]
            
        return B

table=make_table(Conditions,max_f,b)
# main function
print(table)
def simplex(t):
    print('(n,m) = ', (n,m))
    iteraition=0
    while check_neg(t) :
        iteraition+=1
        col=max_neg(t)
        row=min_quot(t,col)
        change_sol(row,col)

        print(iteraition,'IT')
        print('index of values',sol)
        print(row,col,'\npivot element = ',t[row][col])

        t=sumR(t,row,col)
        
        for i in range(m+1):
            if i==0: print('my table:\n')
            print(t[i])
        
    print('F_max = ',b[0])
    for i in range(1,len(sol)+1):

        if sol[i-1]<n:
            print('sol x_',sol[i-1]+1, ' = ', b[i])
        else:
            print('sol S_',sol[i-1]+1, ' = ', b[i])
    for i in range (n):
        if i not in sol:
            print('sol x_',i+1, ' = ', 0)
    print(sol)
simplex(table)
