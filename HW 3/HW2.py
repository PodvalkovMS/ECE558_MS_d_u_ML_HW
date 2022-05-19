from os import link
import pandas as pd

import numpy as np
from collections import defaultdict
import math

from time import sleep
from pandas.tseries.offsets import Second
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
    def printPath(self, parent, j):
        global yolo
        yolo.append(j) # The number of vertex in the path 
        #Base Case : If j is source
        if parent[j] == -1 :
            return 
        
        self.printPath(parent , parent[j])

    
   
    # A utility function to print
    # the constructed distance
    # array
    def printSolution(self, dist, parent):
        global yolo
        src = []
        for i in range(1, len(dist)):
            self.printPath(parent,i)
            src.append(yolo)
            yolo=[]
        return src
       
 
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
 
 
        
        
       

        scr=self.printSolution(dist,parent)                            
        return dist , scr


def SumCostFunction(traffic, dist, parent, task ):  # For a all core calculate the cost of the from left top core to other 
    cost=0
    for i in range(1, len(dist)):
        myneed1=parent[i-1]                                                  # the path from 0 core to i 
        first=task[myneed1[len(myneed1)-1]]                                  # the number of top left core
        second=task[myneed1[0]]                                                      #he number of i core
        cost=cost+CostFunction(traffic[first][second], dist[i], (len(parent[i-1])))
    return cost




def CostFunction(f, l, h ):  # The function of furmula of the cost (m*h+l)*f m=3 h- hops of the shortes path
    cost=(3*h+l)*f           # l the distance witch we get from dijstrak algorithm
    return cost


def ChangeThePlace(a,b): # The function of the swap 2  object use in Tile swap to change the link map, task verctor 
    a1=a                 # and the NoC map.
    b1=b
    
    c=a1
    a1=b1
    b1=c

    return a1, b1




def TileSwap(links, map, task):                 #   Swap a core number in links , Noc and task
    
    links_new=links
    map_new=map
    task_new=task
    
    first_core=np.random.randint(0 ,63)       # Choose to random core to swap
    second_core=np.random.randint(0,63)


    if first_core==second_core:             # Check if it the same core
        if second_core!=63:
            second_core=second_core+1
        else:
            second_core=second_core-1
    
    ftask=myarray(task_new)                            # Find the position of these cores into task the position in task will be
    first=ftask.index(first_core)                     # the same as in links 
    second=ftask.index(second_core)
    
    [task_new[first], task_new[second]]=ChangeThePlace(task_new[first], task_new[second])
    
    [links_new[first,:], links_new[second,:]]=ChangeThePlace(links_new[first,:], links_new[second,:])


    fmap=myarray(map)                                        #Find the the postion of cores in Map 
    [first_i, first_j]=fmap.index(first_core)
    [second_i,second_j]=fmap.index(second_core)

    [map_new[first_i,first_j], map_new[second_i,second_j]]=ChangeThePlace(map_new[first_i,first_j], map_new[second_i,second_j])


    return links_new, map_new, task_new


def ChekingTransferTheBoarder(x, dx):               #Function check if the new link conection conect cores goes throught the boreder 
                                                    # of the 8x8 cores system   
    if x+dx<0: 
        coff=1
    
    else:
        coff=0

    if x+dx>7:
        coff=-1

    else:
        coff=0


    return coff    

def LinkMove(links, map, task):                        # Find the links conection in random core and change it
    
    links_def=links
    random_core=np.random.randint(0 ,64)

    ftask=myarray(task)
    first=ftask.index(random_core)                   # Find the positon of the core in link map and core map

    fmap=myarray(map)
    [first_x, first_y]=fmap.index(random_core)

    flinks=myarray(links[first,:])                            
    [o, op, gold]=flinks.index2(0)                  #Find all links coection to th core 
    if len(gold)>0:                                 # Cheak it if connection 
        end_of_link=np.random.choice(gold)
        links_def[random_core, end_of_link]=0
        links_def[end_of_link, random_core]=0
    else:
        return links_def  



    x_movment=np.random.randint(-3,3)               # Random vertical move in map core
    y_movment=np.random.randint(-math.floor(math.sqrt(16-x_movment**2)),math.floor(math.sqrt(16-x_movment**2)))  # calculate the other lenght that lenght of it will be less or = then 4
    

    coordinate_x_core_new_link=7*ChekingTransferTheBoarder(first_x,x_movment)+first_x+x_movment # Cheak if goes thougth the core border for instane in mesh if it conect 0 core to 7 or 56
    coordinate_y_core_new_link=first_y+y_movment+7*ChekingTransferTheBoarder(first_y,y_movment)

    new_end_of_new_link=map[coordinate_x_core_new_link,coordinate_y_core_new_link]

    links_def[new_end_of_new_link, random_core]=math.ceil(math.sqrt(y_movment**2+x_movment**2))
    links_def[random_core, new_end_of_new_link]=math.ceil(math.sqrt(y_movment**2+x_movment**2))

    for i in range (0, 64): links_def[i,i]=0
    
    return links_def



def SAlgorithm(T,Th, alpha, numlter, task, map, links, traffic_all, g):
    task_plas=task
    map_core=map
    links_core=links
    t=0

    pbar = tqdm(total=((math.ceil(math.log((Th/T), alpha))*numlter)))

    number_of_iteration=0

    while T>=Th:       

        for i in range(numlter):

            pbar.update(1)
            
            number_of_iteration=number_of_iteration+1

            links_core_old=links_core
            map_old=map_core
            task_old=task_plas
            

            [dist, hops]=g.dijkstra(links_core,0)                                    # Calculate the cost function 
            cost_old_links=SumCostFunction(traffic_all, dist, hops, task_plas)
            
            if (np.random.randint(0,2))>0:                                                      # randomly change it
                [link_core_new, map_old, task_old]=TileSwap(links_core_old,map_old ,task_old)
            else:
                link_core_new=LinkMove(links_core_old,map_old, task_old)
            
            [dist, hops]=g.dijkstra(link_core_new,0)                                   # calculate the new cost 
            cost_new_links=SumCostFunction(traffic_all, dist, hops, task_old)
            
            if cost_new_links<cost_old_links:
                links_core=link_core_new
                task_plas=task_old
                map_core=map_old
                t=t+1
            else:
                prob=math.exp(-(cost_new_links-cost_old_links)/T)
                good=np.random.uniform(low=0, high=1)
                if good<prob:
                    links_core=link_core_new
                    task_plas=task_old
                    map_core=map_old
                    t=t+1
        
        
        T=alpha*T

    pbar.close()

    print("\n Number of iteration = \t%d \t\t Number when new links better then the old =\t%d  \n" %(number_of_iteration, t) )
    print('\n')

    return links_core, task_plas, map_core


def CreateEtalonTask():                  # Create a task plasment fot mesh
    garbige=np.zeros(64, dtype=int)
    for i in range(64):
        garbige[i]=i
    return garbige

def CreateEtalonLinks():                          # Create link plasment for mesh
    PALAC=np.zeros((64, 64), dtype=int)

    for i in range(8):
        for j in range(7):
      
            if i!=7:
                PALAC[i+1+j*8, i+j*8]=1
                PALAC[i+j*8, i+1+j*8]=1

            PALAC[(i+8*j) , (i+8*(j+1))]=1
            PALAC[(i+8*(j+1)), (i+8*(j))]=1

      
        if i!=7:
                PALAC[i+1+56, i+56]=1
                PALAC[i+56, i+1+56]=1

    return PALAC

def CreationEtalonMap():                        # Create core map for mesh
    k=0
    garbige=np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            garbige[i,j]=k
            k=k+1
    return garbige


#################################################################################33
# 
# main code 
# 
# ####################################################################################            

# Read the traffic

traffic_complement = pd.read_csv("traffic_complement.csv", header=None) #complement set
traffic_rand = pd.read_csv("traffic_rand.csv", header=None) #rand set
traffic_uniform = pd.read_csv("traffic_uniform.csv", header=None) #uniform set


traffic_matrix=[traffic_complement, traffic_rand, traffic_uniform]

yolo=[]


g=Graph()

name_traffic=['complement', 'rand' , 'uniform']

print('\n')
print('Statr the calculation')
print('\n')

ruper=CreateEtalonLinks()
riper=CreateEtalonTask()

#Put the Mesh into csv

dl=pd.DataFrame(ruper, columns=riper)
dl.to_csv('link_plasment_mesh.csv', encoding='utf-8', index=False)



runger=CreationEtalonMap()


print('\n Initial conditions \n\n Mesh \n')
print(runger)



for i in range(0,3):     # For all traffic do

    T=500              # initite hyper parameters
    Th=0.1
    alpha=0.99
    numlter=100

    print('\n SA algorithm for traffic ' + name_traffic[i])


    TASK_NOW=CreateEtalonTask()
    LINK_NOW=CreateEtalonLinks()
    MAP_NOW=CreationEtalonMap()

    [dist, hops]=g.dijkstra(LINK_NOW,0)
    etalon_cost=SumCostFunction(traffic_matrix[i], dist, hops, TASK_NOW)     # Calculate cost for mesh with traffic


    print("\n Cost of the mesh in traffic "+ name_traffic[i]+ " = \t%d " %( etalon_cost ))
    print("\n")

    TASK=CreateEtalonTask()
    MAP=CreationEtalonMap()
    LINK=CreateEtalonLinks()

    [links_final, task_finals, map_final]=SAlgorithm(T, Th, alpha, numlter, TASK, MAP, LINK, traffic_matrix[i], g) 

    print("\n Final NOC \n")

    print(map_final ,'\n')
    
    print("\n Final Links \n")

    print(links_final)


    [dist, hops]=g.dijkstra(links_final,0)                                      # Calculate the cost of finale result 
    final_cost=SumCostFunction(traffic_matrix[i], dist, hops, task_finals)

    print("\nFinal Cost in traffic "+ name_traffic[i]+ " \t%d " %( final_cost ))

    file_name="Link_plasment_"+name_traffic[i]+".csv"

    dl=pd.DataFrame(links_final, columns=task_finals)                   # Write the result in csv file where data is links placment
    dl.to_csv(file_name, encoding='utf-8', index=False)                  # and the header is task placment 
