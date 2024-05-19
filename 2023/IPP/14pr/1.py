import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
start_time=time.time()
t=100
dt=0.001
Y=5 # for y
dy=1/10
X=10 
dx=1/10
Z=5 
dz=1/10
def subs(L_i,L_j):
    for j in range(X):
        for i in range(Y):
            for k in range(Z):
                if abs(L_i[i][j][k]-L_j[i][j][k])>10**-5:
                    return True     
    print('Steady')
    return False

comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
step_x = rank*(X+1)//size
n_step_x = (rank+1)*(X+1)//size-1
U=np.zeros((t+1,X+1,Y+1,Z+1))
if rank==0: step_x=1
if rank==size-1: n_step_x=X-1
# print(rank,step_x,n_step_x)

b=1
n=0
while b>10**-5:
    if n>t-1:
        if rank==0:
            print("not steady")
        break     
    if rank>0:
        
        comm.Send(U[n][step_x],dest=rank-1,tag=1)
        recv_data = np.full((Y+1,Z+1),0,dtype=float)
        comm.Recv(recv_data,source=rank-1,tag=2)
        U[n][step_x-1]=recv_data   
    if rank<size-1:
        
        comm.Send(U[n][n_step_x],dest=rank+1,tag=2)
        recv_data1 = np.full((Y+1,Z+1),0,dtype=float)
        comm.Recv(recv_data1,source=rank+1,tag=1)
        U[n][n_step_x+1]=recv_data1
    for i in range(X+1):
        for j in range(Y+1):
            for k in range(Z+1):
                
                U[n][0][j][k]=1
                U[n][-1][j][k]=0
                U[n][i][0][k]=0
                U[n][i][-1][k]=0
            U[n][i][j][0]=0
            U[n][i][j][-1]=0    
    for i in range(step_x,n_step_x+1):
        for j in range(1,Y):
            for k in range(1,Z): 
                U[n+1][i][j][k]=(dy*dy*dz*dz *(U[n][i+1][j][k]+U[n][i-1][j][k])+dx*dx*dz*dz*(U[n][i][j+1][k]+U[n][i][j-1][k])+dx*dx*dy*dy*(U[n][i][j][k+1]+U[n][i][j][k-1]))/(2*(dy*dy*dz*dz+dx*dx*dz*dz+dx*dx*dy*dy))
               
                
    n+=1
    
    # diff=comm.allgather(list(abs(U[n][step_x:n_step_x+1]-U[n-1][step_x:n_step_x+1])))
    # # print(abs(U[i][step_x:n_step_x+1]-U[i-1][step_x:n_step_x+1]))
    # # print(f'b = {diff[1][0]}')
    # maximum=[]
    # for k in diff[1:]:
    #     # print(k)
    #     for ind,m in enumerate(k):
    #         for l in m:
    #             maximum.append(max(l))
    # # break
    #     # maximum.append(max(k[0]))

    # b=max(maximum)
        
print(t)
        # b=max(abs(U[i]-U[i-1])) ?????
if rank==0: step_x=0
if rank==size-1: n_step_x=X  
# print(step_x,n_step_x,rank)
# print(rank,U[:,:,step_x:n_step_x+1])
columns = U[:,step_x:n_step_x+1,:, :] 
columns_contiguous = columns.copy(order='C')
# recvcounts = comm.allgather(n_step_x-step_x+1)
# displacements = comm.allgather(step_x)
# if rank==1:
#     print(U[8][step_x:n_step_x+1],rank)
t=comm.gather(U[:,step_x:n_step_x+1,:, :], root=0)

# if rank == 0: print(t[0][0])
# # end_parall=time.time()-start_time
# # start_time_sequential=time.time()
if rank == 0:
    f=[]
    for j in range(X + 1):
            x = list(t[0][j])        
            for i in range(1,size):
                x.extend(t[i][j])
            f.append(x)
    # final=np.concatenate(t,axis=2)
    final=np.array(f)
    # print(final,"final")
    with open("test.txt",'w') as file:
        file.write(str(final))
    # print(len(final))
#     with open('results.txt','a') as file:
    #         file.write('\n'+str(time.time()-start_time))
    # end_seque=time.time()-start_time_sequential
#     print("sequen ", end_seque,rank)
# print('Parral ' ,end_parall,rank)       
# 1.6126201152801514

# S=1.6126201152801514/0.009002685546875
# print(S)
# alpha=(4/3)*(1/S-0.25)
# print(alpha)
# 0.08619523048400879 1 rank
# 0.01600360870361328 4 rank    