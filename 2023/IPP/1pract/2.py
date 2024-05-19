from mpi4py import MPI
n=4
k=2*n
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

cnt_all=0
summ_all=0
for i in range(n):
     if rank ==i:
          summ=summ_all
          cnt=cnt_all
          for j in range(2):
               summ+=((-1)**cnt)/(3**cnt*(2*cnt+1))
               cnt+=1
          print (cnt)     
          print ('summ= ',summ,rank)     
          comm.send(summ,dest=n)
for i in range(n):    
     if rank==n:
          data=comm.recv(source=i)
          print(data, rank)
