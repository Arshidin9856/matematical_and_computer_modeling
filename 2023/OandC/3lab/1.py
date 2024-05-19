import math
import random
import numpy as np
import time
from mpi4py import MPI
i=0
def find_valie(x):
    return math.sin(x[0]+x[1])**2+math.cos(x[1])**2+5-math.e**-((x[0]*x[0]+x[1]*x[1]+146*x[1]+54*x[0]+6058)/25)        
def neg_Gradient(x,y,eps=10**-5):
        
        v_x=(find_valie([x + eps,y]) - find_valie([x - eps,y])) / (2 * eps)
        v_y=(find_valie([x ,y+eps]) - find_valie([x ,y-eps])) / (2 * eps)
        
        v=np.zeros((1,2),dtype=float)
        # c1=math.sin(x+y)*math.cos(x+y)*2
        # c2=math.e**-((x*x+y*y+146*y+54*x+6058)/25)
        # v_x=c1+((2*x+54)/25)*c2
        # v_y=c1-2*math.cos(y)*math.sin(y)+c2*(146+2*y)/25
        v[0][0]=v_x*(-1)
        v[0][1]=v_y*(-1)
        return v
def transp(vector):
    return vector.reshape(2,1)
def backtraking_search():
    t=1
    future_value=find_valie((point+t*direction)[0])
    line=M_value+alpha*t*(np.matmul(direction,transp((-1)*direction)))
    while True:
        if future_value<line[0][0]:
            break
        t=beta*t
        future_value=find_valie((point+t*direction)[0])
        line=M_value+alpha*t*(np.matmul((-direction),transp(direction)))
        if t==0:
            return False
    return t
while i<1001:
    comm = MPI.COMM_WORLD
    size= comm.Get_size()
    rank = comm.Get_rank()
    start_time=time.time()
    start=time.time()
    a,b= 1,1
    limit=False
    if rank==0:
        a=random.random()*100
        b=random.random()*100
        with open('Points.txt','r') as file:
            for x in file:
                while str((round(a),round(b))) == x:
                    a=random.random()*100
                    b=random.random()*100
                    if time.time()-start>10:
                        limit=True
                        break
                if limit:break
        if limit:break
        with open('Points.txt','a') as file:
            file.write('\n'+str((round(a),round(b))))
    buff=np.array([a,b],dtype=float)
        
    comm.Bcast(buff, root =0)
    a=buff[0]
    b=buff[1]
    if rank==0:point=np.array([[a,b]],dtype=float)
    elif rank==1:point=np.array([[-a,-b]],dtype=float)
    elif rank==2:point=np.array([[a,-b]],dtype=float)
    elif rank==3:point=np.array([[-a,b]],dtype=float)


    history=[8,10]
    alpha=0.1
    beta=0.1
    epsilon=10**-50

    M_value=find_valie(point[0])
    direction=neg_Gradient(*point[0])

    while  math.sqrt(direction[0][0]**2+direction[0][1]**2)>epsilon:
        direction=neg_Gradient(*point[0])
        T=backtraking_search()
        if  not T:
            break
        point=point+T*direction
        history.append(M_value)
        M_value=find_valie(point[0])

    with open('Res.txt','a') as file:
            file.write('\n'+str(history[-1]))
    i+=1

print(f'{time.time()-start_time} === time required')

data = np.loadtxt("Res.txt", delimiter='\t', dtype=np.float64)
Final=min(data)
print(Final)
print(point)
Res = np.loadtxt("final.txt", delimiter='\t', dtype=np.float64)
with open('final.txt','a') as file:
            file.write('\n'+str(Final))

if Final<min(Res):
    print('\n', f'found smaller {Final} at {point}')
    with open('final_res.txt','a') as file:
                file.write('\n'+str(Final)+f' {rank} =rank')
                file.write('\n'+str(point)+f' {rank} =rank')

    # f((-26.753420918968843, -73.7871437093669)) = 4.028562890611684 
# print(4.028562890611684>4.028562890609468)