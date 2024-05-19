from mpi4py import MPI
import math
import numpy as np

comm = MPI.COMM_WORLD
n= comm.Get_size()
rank = comm.Get_rank()
cnt_all=0
summ_all=np.zeros(1)
summ=np.zeros(1)  

for i in range(rank,10*n,n):
    # print(summ)
    summ+=((-1)**i)/(3**i*(2*i+1))
    comm.Reduce(summ,summ_all,op=MPI.SUM,root=0)
    # print(i,'= index rank =',rank,'summ= ',summ)
         
if rank == 0 :
        print(summ_all*2*math.sqrt(3))
               
