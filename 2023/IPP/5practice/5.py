from mpi4py import MPI
# for N CPU
import random
vector=[15,156,164,121,64,133,461,21,156666666,12,1,0,12,31,21,1131,2,3]

N=len(vector)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
max=vector[rank]
print(rank,max)
l_max=[]    
if rank!=size-1:
    for i in range(rank,N,size-1):
        # print('HERE',rank,i)
        if vector[i]>max:
            max=vector[i]
    comm.send(max,dest=size-1)
elif rank==size-1:
    for i in range(size-1):
        max1=comm.recv(source=i)
        l_max.append(max1)
    maximal=l_max[0]        
    for i in range(size-1):
        if maximal<l_max[i]:
            maximal=l_max[i]
        
    print(maximal ,'from size-1 CPU')
# # for MIN
min=vector[rank]
l_min=[]    
if rank!=size-1:
    for i in range(rank,N,size-1):
        # print('HERE',rank,i)
        if vector[i]<min:
            min=vector[i]
    comm.send(min,dest=size-1)
elif rank==size-1:
    for i in range(size-1):
        min1=comm.recv(source=i)
        l_min.append(min1)
    minimal=l_min[0]        
    for i in range(size-1):
        if minimal>l_min[i]:
            minimal=l_min[i]
        
    print(minimal ,'from size-1 CPU')
        