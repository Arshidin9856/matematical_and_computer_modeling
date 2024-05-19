import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
start_time=time.time()
t=8
N=5 # for y
dy=1/100
n=5 
dx=1/100

comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
step_x = rank*(n+1)//size
n_step_x = (rank+1)*(n+1)//size-1
U=np.zeros((t+1,n+1,N+1),dtype='float64')
if rank==0: step_x=1
if rank==size-1: n_step_x=n-1
b=1
i=1
U[0]=np.full((n+1,N+1),8)
while b>10**-2:
    if i>t-1:
        if rank==0:
            print("not steady")
        break     
    if rank>0:
        
        comm.Send(U[i][step_x],dest=rank-1,tag=1)
        recv_data = np.full((1,n+1),0 ,dtype=float)
        comm.Recv(recv_data,source=rank-1,tag=2)
        U[i][step_x-1]=recv_data   
    if rank<size-1:
        
        comm.Send(U[i][n_step_x],dest=rank+1,tag=2)
        recv_data1 = np.full((1,n+1),0, dtype=float)
        comm.Recv(recv_data1,source=rank+1,tag=1)
        U[i][n_step_x+1]=recv_data1
    for k in range(1,n):
        U[i][0][k]=1
        U[i][n][k]=0
        for j in range(step_x,n_step_x+1):
            U[i][j][0]=0
            U[i][j][n]=0    
            U[i+1][j][k]=1/4*((dx**2)*(U[i][j][k+1]+U[i][j][k-1])+(dy**2)*(U[i][j-1][k]+U[i][j+1][k]))    
    i+=1
    # b=max(abs(U[i]-U[i-18])) ?????
if rank==0: step_x=0
if rank==size-1: n_step_x=n    
columns = U[:, step_x:n_step_x+1]
columns_contiguous = columns.copy(order='C')
# recvcounts = comm.allgather(n_step_x-step_x+1)
# displacements = comm.allgather(step_x)
t=comm.gather(columns_contiguous, root=0)
end_parall=time.time()-start_time
start_time_sequential=time.time()
if rank == 0:
    f=[]
    for j in range(N+1):
        x=list(t[0][j])
        for i in range(1,size):
            x.extend(t[i][j])
        f.append(x)
    final=np.array(f)
    print(final)
#     with open('results.txt','a') as file:
#             file.write('\n'+str(time.time()-start_time))
#     end_seque=time.time()-start_time_sequential
#     print("sequen ", end_seque,rank)
print('Parral ' ,end_parall,rank)           