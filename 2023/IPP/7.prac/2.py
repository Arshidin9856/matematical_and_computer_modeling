from mpi4py import MPI
import math
import numpy as np

unordered_list=[100,7,10,2,46]
comm = MPI.COMM_WORLD
size= comm.Get_size()
rank = comm.Get_rank()
N=len(unordered_list)
res=[]
sub_list=[]
b=[]
for i in range(size):
    b.append(False)
stop=True
def all_T(arr):
    for i in range(len(arr)):
        if i ==False: return False
H=-1
while  not all_T(b):
    H+=1
    sub_list.append([])
    for i in range (rank*N//size,(rank+1)*N//size,1):    
        sub_list[H].append(unordered_list[i])
    if len(sub_list[H])<=2:
        b[rank]=True
    else: