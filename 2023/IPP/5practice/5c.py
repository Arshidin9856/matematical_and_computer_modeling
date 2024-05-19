from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



if (rank != 0):
    data=comm.recv(source=rank-1)
    print(f"Process {rank} received value {data} from process {rank-1}\n")
cnt=0

comm.send(cnt,dest=((rank+1)%size))

if (rank == 0) :
    data=comm.recv(source=size - 1)
    print(f"Process {rank} received value {data} from process {size-1}\n")
