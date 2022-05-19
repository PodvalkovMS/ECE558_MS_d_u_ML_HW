from os import link
import pandas as pd

import numpy as np
from collections import defaultdict
import math

from time import sleep
from pandas.tseries.offsets import Second
from tqdm import tqdm
from sklearn import  linear_model

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

    new_end_of_new_link=int(map[coordinate_x_core_new_link,coordinate_y_core_new_link])

    links_def[new_end_of_new_link, random_core]=math.ceil(math.sqrt(y_movment**2+x_movment**2))
    links_def[random_core, new_end_of_new_link]=math.ceil(math.sqrt(y_movment**2+x_movment**2))

    for i in range (0, 64): links_def[i,i]=0
    
    return links_def


def FindTheCost(links, traffic, task):
    g=Graph()
    [dist, hops]=g.dijkstra(links,0)
    cost=SumCostFunction(traffic, dist, hops, task)
    return cost


# hill climbing local search algorithm
def hillclimbing(objective, links, map_f,  task_1, traffic, n_iterations):
	# generate an initial point
    solution_links=links
    solution_task=task_1
    solution_map=map_f
    i=0
    all_solution_links=np.empty((n_iterations, 64,64), dtype=object)
    all_solution_map=np.empty((n_iterations, 8, 8), dtype=object)
    all_solution_task=np.empty((n_iterations, 64), dtype=object)
    all_solution_cost=[]
    stop_criteria=0
	# evaluate the initial point
    solution_eval = objective(solution_links, traffic, solution_task)

    all_solution_links[0]= solution_links
    all_solution_map[0]= solution_map
    all_solution_task[0]= solution_task
    all_solution_cost.append(solution_eval)

	# run the hill climb
    while (i<n_iterations-1) :
		# take a step
        if (np.random.randint(0,2))>0:                                                      # randomly change it
            [candidate_core, candidate_map, candidate_task]=TileSwap(solution_links ,solution_map , solution_task)
        else:
            candidate_core=LinkMove(solution_links ,solution_map , solution_task)
            candidate_map, candidate_task=solution_map , solution_task
		# evaluate candidate point
        candidte_eval = objective(candidate_core, traffic, candidate_task)
		# check if we should keep the new point

        if int(candidte_eval) <=int(solution_eval):
			# store the new point
            stop_criteria=0
            solution_links, solution_eval, solution_map, solution_task = candidate_core, candidte_eval, candidate_map ,candidate_task
            all_solution_links[i+1]= solution_links
            all_solution_map[i+1]= solution_map
            all_solution_task[i+1]= solution_task
            all_solution_cost.append(solution_eval)
            i=i+1
			# report progress
            print('>%d  = %.5f' % (i, solution_eval))


    return all_solution_links, all_solution_map, all_solution_task, all_solution_cost

def ChangeMatrixToFitForm(links, tasks):

    a=links[np.triu_indices(64, k = 1)]
    b=tasks
    input_links=np.concatenate((a, b), axis=None)
    input_links.reshape(-1, 2080)


    return input_links

def ChangeAllMatrixFitTrain(links, tasks):
    k=len(links)
    train_X=np.empty((k, 2080) ,dtype=object)
    for i in range(len(links)):
        train_X[i]=ChangeMatrixToFitForm(links[i],tasks[i] )


    return train_X

def CostOfRegressor(core, traffic, task):
    global regressor
    Predicton_data=ChangeMatrixToFitForm(core, task)
    cost=regressor.predict(Predicton_data.reshape(-1, 2080))

    return cost

def STAGEAlgorithm(T, task, map, links, traffic_all, g):
    global regressor
    kalichestvo=int((len(links)**2-len(links))/2)+len(task)
    train_X=np.empty((T, kalichestvo) ,dtype=object)
    Cost_first_links=FindTheCost(links, traffic_all, task)
    smotritel=0
    glavnoe=0
    LINKS_TRASH=links
    MAPS_TRASH=map
    TASK_TRASH=task
    proverka=0
    gaga=0
    pbar = tqdm(total=10)

    while glavnoe<5:
        proverka=0
        gaga=gaga+1
        [trajectory_links, trajector_map, trajectory_task, all_cost]=hillclimbing(FindTheCost, LINKS_TRASH, MAPS_TRASH, TASK_TRASH, traffic_all, T)
        
        #pbar.update(1)

        train_Y=np.full(T, min(all_cost))

        train_X=ChangeAllMatrixFitTrain(trajectory_links,trajectory_task)
        regressor.fit(train_X, train_Y )
        S0_new_index=np.random.randint(1,T-1)
        while proverka<1:
        
            

            [trajectory_links_new, trajector_map_new, trajectory_task_new, all_cost_new]=hillclimbing(CostOfRegressor, trajectory_links[S0_new_index], trajector_map[S0_new_index], trajectory_task[S0_new_index], traffic_all, T)
    
            final_design_links=trajectory_links_new[len(trajectory_links_new)-1]
            final_design_task=trajectory_task_new[len(trajectory_task_new)-1]
            final_design_map=trajector_map_new[len(trajector_map_new)-1]
            final_design_cost=all_cost_new[len(all_cost_new)-1]

            if np.array_equal(trajectory_links_new[len(trajectory_links_new)-1],trajectory_links[len(trajectory_links)-1]):
                S0_new_index=np.random.randint(1,T-1)
                smotritel=smotritel+1
                if smotritel>3:
                    return final_design_links, final_design_map, final_design_task
            else:
                smotritel=0
                proverka=1
                LINKS_TRASH=trajectory_links[S0_new_index]
                MAPS_TRASH=trajector_map[S0_new_index]
                TASK_TRASH=trajectory_task[S0_new_index]
        if abs((min(all_cost)-min(all_cost_new)))/min(min(all_cost),min(all_cost_new))<0.05:
            glavnoe=glavnoe+1




    

    return final_design_links, final_design_map, final_design_task, final_design_cost



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

regressor= linear_model.LinearRegression()

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



for i in range(0,1):     # For all traffic do

    T=100        # initite hyper parameters
 

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

    [links_final,  map_final, task_finals, final_cost]=STAGEAlgorithm(T, TASK, MAP, LINK, traffic_matrix[i], g) 

    print("\n Final NOC \n")

    print(map_final ,'\n')
    
    print("\n Final Links \n")

    print(links_final)



    print("\nFinal Cost in traffic "+ name_traffic[i]+ " \t%d " %( final_cost ))

    file_name="Link_plasment_"+name_traffic[i]+".csv"

    dl=pd.DataFrame(links_final, columns=task_finals)                   # Write the result in csv file where data is links placment
    dl.to_csv(file_name, encoding='utf-8', index=False)                  # and the header is task placment 
