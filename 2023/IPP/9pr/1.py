from mpi4py import MPI
import math
import numpy as np
dt=1/10
dx=0.5
n=10
N=10

comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
def communic(ROW):
    if rank>0:
        comm.Send(ROW[step_x],dest=rank-1,tag=1)
        recv_data = np.full(1,0 ,dtype=float)
        comm.Recv(recv_data,source=rank-1,tag=2)
        ROW[step_x-1]=recv_data[0]   
    if rank<size-1:
        comm.Send(ROW[n_step_x],dest=rank+1,tag=2)
        recv_data1 = np.full(1,0, dtype=float)
        comm.Recv(recv_data1,source=rank+1,tag=1)
        ROW[n_step_x+1]=recv_data1[0]
    
step_x = rank*(n+1)//size
n_step_x = (rank+1)*(n+1)//size-1
T=np.zeros((N+1,n+1),dtype=float)

if rank==0: step_x=1
if rank==size-1: n_step_x=n-1
# (8, 10) 3
# (2, 4) 1
# (5, 7) 2
# (0, 1) 0
# print((step_x,n_step_x),rank)
for i in range(N):
    if rank==0:
        T[i][0]=1
    if rank==size-1:
        T[i][n]=0
    communic(T[i])
    # if rank>0:
    #     comm.Send(T[i][step_x],dest=rank-1,tag=1)
    #     recv_data = np.full(1,0 ,dtype=float)
    #     comm.Recv(recv_data,source=rank-1,tag=2)
    #     T[i][step_x-1]=recv_data[0]   
    # if rank<size-1:
    #     comm.Send(T[i][n_step_x],dest=rank+1,tag=2)
    #     recv_data1 = np.full(1,0, dtype=float)
    #     comm.Recv(recv_data1,source=rank+1,tag=1)
    #     T[i][n_step_x+1]=recv_data1[0]
    for j in range(step_x,n_step_x+1):
            T[i+1][j]=(dt/(dx**2))*(T[i][j+1]-2*T[i][j]+T[i][j-1])+T[i][j]
            # print(rank,j)
if rank==0: step_x=0
if rank==size-1: n_step_x=n

columns = T[:, step_x:n_step_x+1]
columns_contiguous = columns.copy(order='C')
# recvcounts = comm.allgather(n_step_x-step_x+1)
# displacements = comm.allgather(step_x)

t=comm.gather(columns_contiguous, root=0)

if rank == 0:
    # Extract the relevant parts from the gathered data
    f=[]
    for j in range(N+1):
        x=list(t[0][j])
        for i in range(1,size):
            x.extend(t[i][j])
        f.append(x)
    final=np.array(f)
    print(final)

