#Insert sort method.
#Selection sort method.
def selection_sort(unordered_list):
#The function takes the list to be sorted as a parameter.
    for idx in range(len(unordered_list)):
#We take the position 'idx' for the lenght of the list.
        min_idx = idx
#We assume the position of the smallest element is idx hence min_idx.
        for j in range(idx+1, len(unordered_list)):
            if unordered_list[min_idx] > unordered_list[j]:
#We check if it is indeed the smallest number and record its position
                min_idx = j
#We swap the elements so that they can be in order
        unordered_list[idx], unordered_list[min_idx] = unordered_list[min_idx], unordered_list[idx]
# selection_sort(unordered_list)
# print(unordered_list)
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

for i in range (rank*N//size,(rank+1)*N//size,1):    
    sub_list.append(unordered_list[i])
# print(len(unordered_list),sub_list,rank)

if len(sub_list)==1:
    if len(res)!=0:
        if  res[len(res)-1]<sub_list[0]:
            res.append(sub_list[0])
        else: 
            temp=res
            res=sub_list
            sub_list.extend(temp)
            res=sub_list

        
    else:
        res.append(sub_list[0])
        # comm.Bcast(res,root=rank)

else:
    print(res,rank)
    for idx in range(len(sub_list)):
        min_idx = idx
        for j in range(idx+1, len(sub_list)):
            if sub_list[min_idx] > sub_list[j]:
                min_idx = j
        sub_list[idx], sub_list[min_idx] = sub_list[min_idx], sub_list[idx]
    if len(res)!=0:
        i,j,k = 0 
        m=len(sub_list)
        n=len(res)
        temp_res=np.zeros(n+m)
        while (i < m and j < n) :
            if (sub_list[i] < res[j]) :
                temp_res[k]=sub_list[i]
                i+=1
            else:
                temp_res[k]=res[j]
                j+=1
            
            k+=1
        while (i < m):
            temp_res[k] = sub_list[i]
            k+=1
            i+=1
        
    
        while (j < n): 
            temp_res[k] = res[j]
            k+=1
            j+=1
    
        for l in range(m):
            sub_list[l] = temp_res[l]
    
        for l in range(m,m+n):
            res[i - m] = temp_res[i]
        print(rank,res,temp_res,sub_list)    
    

    
    
    
    else:res.extend(sub_list)