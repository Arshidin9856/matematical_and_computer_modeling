import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
start_time=time.time()
N=84
dt=1/100
n=10
dx=1/100

comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
step_x = rank*(n+1)//size
n_step_x = (rank+1)*(n+1)//size-1
U=np.zeros((N+1,n+1),dtype=float)
U[0]=np.full(n+1,5)
U[N][0]=1
if rank==0: step_x=1
if rank==size-1: n_step_x=n-1
b=1
i=0
while b>10**-3 :
    if i>N-1:
        if rank==0:
            print("not steady")
        break     
    if rank==0:U[i][0]=1
    if rank==size-1:U[i][n]=0
    if rank>0:
        comm.Send(U[i][step_x],dest=rank-1,tag=1)
        recv_data = np.full(1,0 ,dtype=float)
        comm.Recv(recv_data,source=rank-1,tag=2)
        U[i][step_x-1]=recv_data[0]   
    if rank<size-1:
        comm.Send(U[i][n_step_x],dest=rank+1,tag=2)
        recv_data1 = np.full(1,0, dtype=float)
        comm.Recv(recv_data1,source=rank+1,tag=1)
        U[i][n_step_x+1]=recv_data1[0]
    for j in range(step_x,n_step_x+1):
        U[i+1][j]=(U[i][j+1]-U[i][j-1])/2    
    i+=1
     #?????
    b=comm.allgather(list(abs(U[i][step_x:n_step_x+1]-U[i-1][step_x:n_step_x+1])))
    # print(f'b = {b}')
    maximum=[]
    for k in b:
        maximum.append(max(k))
    b=max(maximum)
if rank==0: step_x=0
if rank==size-1: n_step_x=n    
# print(U)
# print(rank,U)
columns = U[:, step_x:n_step_x+1]
columns_contiguous = columns.copy(order='C')
# recvcounts = comm.allgather(n_step_x-step_x+1)
# displacements = comm.allgather(step_x)
t=comm.gather(columns_contiguous, root=0)
if rank == 0:
    # Extract the relevant parts from the gathered data
    f=[]
    for j in range(N+1):
        x=list(t[0][j])
        for q in range(1,size):
            x.extend(t[q][j])
        f.append(x)
    final=np.array(f)
    print(final,f"at {i} iter")
    with open('results.txt','a') as file:
            file.write('\n'+str(time.time()-start_time))
end_parall=time.time()-start_time
print('Parral ' ,end_parall,rank)