from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


partner_rank = (rank + 1) % 2
for cnt in range(101):
    if (rank == cnt % 2):
        comm.send(cnt, dest= partner_rank)
        print(f"{rank} sent and incremented cnt {cnt} to {partner_rank}\n")
    else:
        data=comm.recv(source=  partner_rank)
        print(f"{rank} received cnt {cnt} from {partner_rank}\n")
    