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
