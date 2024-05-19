from mpi4py import MPI
import math
import numpy as np

comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
# A=np.zeros((5,5))
# B=np.zeros((5,5))  
A=np.array([[0,1,3],
    [2,3,4]],dtype='float64')            
B=np.array([[2,1,1],
    [2,3,1]],dtype='float64')      
n=len(A)*len(A.transpose())
step=n//size
if rank==0:
    res=np.zeros(n,dtype='float64')
else: res=np.empty(n,dtype='float64')


# print(rank,n,len(loc_res))
a=[]
b=[]
for i in range(len(A)):
    for j in range(len(A[i])):
            a.append(A[i][j])
for i in range(len(B)):
        for j in range(len(B[i])):
            b.append(B[i][j])
a=np.array(a)
b=np.array(b)

if n/size!=step and rank==size-1 :
     loc_res=np.empty(step+1,dtype='float64')
    #  loc_a=
else :loc_res=np.empty(step,dtype='float64')
print(a,b)

if size==1:
    for i in range(len(a)):
        res[i]=a[i]+b[i]
else: 
    for i in range(rank*(step),(rank+1)*step,1):
        summ=a[i]+b[i]
        loc_res[i]=summ
    res_loc=comm.gather(loc_res,root=0)
    comm.Barrier()

print(rank,res)       