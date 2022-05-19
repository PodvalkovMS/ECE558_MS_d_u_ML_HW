from os import link
import pandas as pd

import numpy as np
from collections import defaultdict
import math

from time import sleep
from tqdm import tqdm

class myarray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(myarray)
    def index(self, value):
        return np.where(self == value)
    def index2(self, value):
        return np.where(self >value)




#Class to represent a graph
class Graph:
 
    # A utility function to find the
    # vertex with minimum dist value, from
    # the set of vertices still in queue
    def minDistance(self,dist,queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1
         
        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index
 
 
    # Function to print shortest path
    # from source to j
    # using parent array
    def printPath(self, parent, j, i):
        global yolo
        yolo=yolo+1
        #Base Case : If j is source
        if parent[j] == -1 :
            return 
        
        self.printPath(parent , parent[j], i)

    
         
 
    # A utility function to print
    # the constructed distance
    # array
    
 
    '''Function that implements Dijkstra's single source shortest path
    algorithm for a graph represented using adjacency matrix
    representation'''
    def dijkstra(self, graph, src):
        global yolo
        row = len(graph)
        col = len(graph[0])
 
        # The output array. dist[i] will hold
        # the shortest distance from src to i
        # Initialize all distances as INFINITE
        dist = [float("Inf")] * row
 
        #Parent array to store
        # shortest path tree
        parent = [-1] * row
 
        # Distance of source vertex
        # from itself is always 0
        dist[src] = 0
     
        # Add all vertices in queue
        queue = []
        for i in range(row):
            queue.append(i)
             
        #Find shortest path for all vertices
        while queue:
 
            # Pick the minimum dist vertex
            # from the set of vertices
            # still in queue
            u = self.minDistance(dist,queue)
 
            # remove min element    
            queue.remove(u)
 
            # Update dist value and parent
            # index of the adjacent vertices of
            # the picked vertex. Consider only
            # those vertices which are still in
            # queue
            for i in range(col):
                '''Update dist[i] only if it is in queue, there is
                an edge from u to i, and total weight of path from
                src to i through u is smaller than current value of
                dist[i]'''
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u
 
 
        # print the constructed distance array
        
       

        self.printPath( parent, len(dist)-1,  -1)
        path_tom=yolo-1
        yolo=0
        return dist , path_tom






def CostFunction(f, l, h ):
    cost=(3*h+l)*f
    return cost


def ChangeThePlace(a,b):
    a1=a
    b1=b
    
    c=a1
    a1=b1
    b1=c

    return a1, b1




def TileSwap(links, map, task):
    
    links_new=links
    map_new=map
    task_new=task
    
    first_core=np.random.randint(0 ,63)
    second_core=np.random.randint(0,63)


    if first_core==second_core:
        if second_core!=63:
            second_core=second_core+1
        else:
            second_core=second_core-1
    
    ftask=myarray(task_new)
    first=ftask.index(first_core)
    second=ftask.index(second_core)
    
    [task_new[first], task_new[second]]=ChangeThePlace(task_new[first], task_new[second])
    
    [links_new[first,:], links_new[second,:]]=ChangeThePlace(links_new[first,:], links_new[second,:])


    fmap=myarray(map)
    [first_i, first_j]=fmap.index(first_core)
    [second_i,second_j]=fmap.index(second_core)

    [map_new[first_i,first_j], map_new[second_i,second_j]]=ChangeThePlace(map_new[first_i,first_j], map_new[second_i,second_j])


    return links_new, map_new, task_new


def ChekingTransferTheBoarder(x, dx):
    
    if x+dx<0: 
        coff=1
    
    else:
        coff=0

    if x+dx>7:
        coff=-1

    else:
        coff=0


    return coff    

def LinkMove(links, map, task):
    
    links_def=links
    random_core=np.random.randint(0 ,64)

    ftask=myarray(task)
    first=ftask.index(random_core)

    fmap=myarray(map)
    [first_x, first_y]=fmap.index(random_core)

    flinks=myarray(links[first,:])
    [o, op, gold]=flinks.index2(0)
    if len(gold)>1:
        end_of_link=np.random.choice(gold)
        links_def[random_core, end_of_link]=0
        links_def[end_of_link, random_core]=0



    x_movment=np.random.randint(-3,3)
    y_movment=np.random.randint(-math.floor(math.sqrt(16-x_movment^2)),math.floor(math.sqrt(16-x_movment^2))+1)
    

    coordinate_x_core_new_link=7*ChekingTransferTheBoarder(first_x,x_movment)+first_x+x_movment 
    coordinate_y_core_new_link=first_y+y_movment+7*ChekingTransferTheBoarder(first_y,y_movment)

    new_end_of_new_link=map[coordinate_x_core_new_link,coordinate_y_core_new_link]

    links_def[new_end_of_new_link, random_core]=math.ceil(math.sqrt(y_movment**2+x_movment**2))
    links_def[random_core, new_end_of_new_link]=math.ceil(math.sqrt(y_movment**2+x_movment**2))
    
    return links_def



def SAlgorithm(T,Th, alpha, numlter, task, map, links, traffic_all, g):
    task_plas=task
    map_core=map
    links_core=links
    t=0

    pbar = tqdm(total=84800)

    number_of_iteration=0

    while T>=Th:

        for i in range(numlter):

            pbar.update(1)
            
            number_of_iteration=number_of_iteration+1

            links_core_old=links_core

            
            jndex_last=map_core[7,7]
            index_last=map_core[0,0]

            
            traffic=traffic_all[index_last][jndex_last]

            [dist, numbr_of_hops]=g.dijkstra(links_core,0)
            cost_old_links=CostFunction(traffic, dist[63], numbr_of_hops)
            
            if (np.random.randint(0,2))>0:
                [link_core_new, map_core, task_plas]=TileSwap(links_core_old,map_core,task_plas)
            else:
                link_core_new=LinkMove(links_core_old,map_core,task_plas)
            
            [dist, numbr_of_hops]=g.dijkstra(link_core_new,0)
            cost_new_links=CostFunction(traffic, dist[63], numbr_of_hops)
            
            if cost_new_links<cost_old_links:
                links_core=link_core_new
                t=t+1
            else:
                prob=math.exp(-(cost_new_links-cost_old_links)/T)
                good=np.random.uniform(low=0.6, high=1)
                if good<prob:
                    links_core=link_core_new
                    t=t+1
        
        
        T=alpha*T

    pbar.close()

    print("\n Number of iteration = \t%d \t\t Number when new links better then the old =\t%d  \n" %(number_of_iteration, t) )
    print('\n')

    return links_core, task_plas, map_core




#################################################################################33
# 
# main code 
# 
# ####################################################################################            



traffic_complement = pd.read_csv("traffic_complement.csv", header=None) #complement set
traffic_rand = pd.read_csv("traffic_rand.csv", header=None) #rand set
traffic_uniform = pd.read_csv("traffic_uniform.csv", header=None) #uniform set


traffic_matrix=[traffic_complement, traffic_rand, traffic_uniform]

yolo=0

task_plasment=np.zeros(64, dtype=int)
for i in range(64):
    task_plasment[i]=i




core_map=np.zeros((8, 8), dtype=int)


k=0


link_placment=np.zeros((64, 64), dtype=int)
for i in range(8):
    for j in range(7):
      
        if i!=7:
            link_placment[i+1+j*8, i+j*8]=1
            link_placment[i+j*8, i+1+j*8]=1

        link_placment[(i+8*j) , (i+8*(j+1))]=1
        link_placment[(i+8*(j+1)), (i+8*(j))]=1
      
    if i!=7:
            link_placment[i+1+56, i+56]=1
            link_placment[i+56, i+1+56]=1

for i in range(8):
    for j in range(8):
        core_map[i,j]=k
        k=k+1


g= Graph()

name_traffic=['complement', 'rand' , 'uniform']

print('\n')
print('Statr the calculation')
print('\n')


dl=pd.DataFrame(link_placment, columns=task_plasment)
dl.to_csv('link_plasment_mesh.csv', encoding='utf-8', index=False)


etalon_cost=task_plasment
etalon_map=core_map
etalon_link_placment=link_placment



print('\n Initial conditions \n\n Mesh \n')
print(etalon_map)



for i in range(0,3):

    T=500
    Th=0.1
    alpha=0.99
    numlter=100

    print('\n SA algorithm for traffic ' + name_traffic[i])

    first_core=core_map[0,0]
    last_core=core_map[7,7]

    traffic=traffic_complement[first_core][last_core]

    [dist, numbr_of_hops]=g.dijkstra(etalon_link_placment,0)
    etalon_cost=CostFunction(traffic, dist[63], numbr_of_hops)

    print("\n Cost of the mesh in traffic "+ name_traffic[i]+ " = \t%d " %( etalon_cost ))
    print("\n")

    [links_final, task_finals, map_final]=SAlgorithm(T, Th, alpha, numlter, task_plasment, core_map, link_placment, traffic_matrix[i], g) 

    print("\n Final NOC \n")

    print(map_final ,'\n')
    


    first_core=map_final[0,0]
    last_core=map_final[7,7]

    traffic=traffic_complement[first_core][last_core]

    [dist, numbr_of_hops]=g.dijkstra(links_final,0)
    final_cost=CostFunction(traffic, dist[63], numbr_of_hops)

    print("\nFinal Cost in traffic "+ name_traffic[i]+ " \t%d " %( final_cost ))

    file_name="Link_plasment_"+name_traffic[i]+".csv"

    dl=pd.DataFrame(links_final, columns=task_finals)
    dl.to_csv(file_name, encoding='utf-8', index=False)

