import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
start_time=time.time()
t=10000
dt=0.001
N=5 # for y
dy=1/10
n=5 
dx=1/10
def subs(L_i,L_j):
    
    for i in range(N):
        for j in range(n):
            if abs(L_i[i][j]-L_j[i][j])>10**-5:
                return True     
    print('Steady')
    return False

comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
step_x = rank*(n+1)//size
n_step_x = (rank+1)*(n+1)//size-1
U=np.zeros((t+1,n+1,N+1),dtype=np.double)
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
        recv_data = np.full((1,n+1),0 ,dtype=np.double)
        comm.Recv(recv_data,source=rank-1,tag=2)
        U[i][step_x-1]=recv_data   
    if rank<size-1:
        
        comm.Send(U[i][n_step_x],dest=rank+1,tag=2)
        recv_data1 = np.full((1,n+1),0, dtype=np.double)
        comm.Recv(recv_data1,source=rank+1,tag=1)
        U[i][n_step_x+1]=recv_data1
    for k in range(1,n):
        U[i][0][k]=1
        U[i][n][k]=0
        for j in range(step_x,n_step_x+1):
            U[i][j][0]=0
            U[i][j][n]=0    
            U[i+1][j][k]=dt/(dx*dx)*(U[i][j+1][k]-2*U[i][j][k]+U[i][j-1][k])+dt/(dy*dy)*(U[i][j][k+1]-2*U[i][j][k]+U[i][j][k-1])+ U[i][j][k]    
    i+=1
    if size==1:
     b=subs(U[i],U[i-1]) 
     if not b :
         break
    else:     
        b=comm.allgather(list(abs(U[i][step_x:n_step_x+1]-U[i-1][step_x:n_step_x+1])))
        # print(abs(U[i][step_x:n_step_x+1]-U[i-1][step_x:n_step_x+1]))
        # print(f'b = {b}')
        maximum=[]
        for k in b[1:]:
            # print(k)
            maximum.append(max(k[0]))
            # maximum.append(max(k[0]))

        b=max(maximum)
        # b=max(abs(U[i]-U[i-18])) ?????
print(i)
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
# 1.6126201152801514

S=1.6126201152801514/0.009002685546875
print(S)
alpha=(4/3)*(1/S-0.25)
print(alpha)
# 0.08619523048400879 1 rank
# 0.01600360870361328 4 rank    