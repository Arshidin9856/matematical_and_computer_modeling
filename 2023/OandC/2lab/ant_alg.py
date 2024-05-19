import math
import random
import numpy as np
import time
start_time=time.time()
N=194
iter=0
num_it=10
ANT_num=100
Q=10

rho=0.1 # preromone evoporation rate
alpha=1 # influence of pher
beta=5# influence of dist

def find_cost(path):
    y=0
    for i in range(N-1):
        y+=dist[path[i]][path[i+1]]
    return y        

def path_ant(Best_t,Best_c):
    pher=np.copy(pherom)
    path=[0]
    pher=pher.tolist()
    # creating path
    while len(path)<N:
        try:
            cnt=path[-1]
            values=[i for i in range(N) if i not in path and dist[cnt][i]>0]
            if len(values)==0:
                 return Best_t,Best_c
            p=[A[cnt][j] for j in values]
            x=random.choices(values,p)[0]
            path.append(x)
            
        except ValueError:
            print('start',iter,'\n',(cnt,path[N-1], path[cnt]),'\n',pherom[path[cnt]-1],'\nEnd')
    path.append(0)

    y= find_cost(path)
    
    history.append(y)
    # feedback depends on len
    
    for i in range(1,N):
        pherom[(path[i-1],path[i])]+=Q/y
    if y<Best_c or Best_c==0:
        print("found better", y,iter)
        for i in range(1,N):
            pherom[(path[i-1],path[i])]+=Q/y
        return path,y
    
    # else: print(y,iter)
    return Best_t,Best_c
coordinates=[]
with open('qa194.txt', 'r') as file:
        for line in file:
            z,x, y = map(float, line.split())
            coordinates.append((x, y))

dist=np.zeros((N,N),int)
for i in range(N):
    for j in range(i,N): 
        dist[j][i]=dist[i][j] = math.floor(np.sqrt((coordinates[i][0]-coordinates[j][0])**2+(coordinates[i][1]-coordinates[j][1])**2)+0.5)  

eta=1/ (dist+1e-10)
# pherom depends on distance 
A=np.full((N,N),0.1)
# i use A to choose path
pherom=np.zeros((N,N),float)
# pherom=np.full((N,N),1.)
for i in range(N):            

    for j in range(i,N):
            if i==j: pherom[i][i]=0
            else: 
                x = (np.amax(dist) - dist[i][j]) / np.amax(dist)-np.amin(dist) * 1 + 0.1
                while x==0 :
                    x=0.1 
                pherom[i][j]=x
                pherom[j][i]=pherom[i][j]

history=[]
Best_cost=0
Best_tour=[]
while  iter<num_it or not Best_cost!=9352:
    A=(pherom**alpha)*(eta**beta) 
    if iter>0: pherom*=(1-rho)
    pherom=np.round(pherom,2)
    # Ants_tours=np.zeros((ANT_num,N+1),int)
    for i in range(ANT_num):
        # path_ant changes pherom and give lyn for every ant
        # Ants_tours[i],
        Best_tour,Best_cost=path_ant(Best_tour,Best_cost)
        # feedback for elite tour
    print(iter)
    iter+=1
print(Best_tour,Best_cost,'Final')
with open('Optimal_best.txt','a') as file:
        file.write('\n'+str(Best_cost))
print(time.time()-start_time)
with open('Optimal_bestT.txt','w') as file:
        file.write(str(Best_tour))
