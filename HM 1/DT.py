import pandas as pd
import math
import numpy as np
from random import randint
import copy

#Evaluates a sample using the given decision tree dt. Returns true if the decision tree gets it right, false if not
def evalSample(dt, sample):   
    while (len(dt) == 2):
        dt = dt[1][sample[dt[0]]]
    return (dt[0] == sample['party'])

infileTrain = pd.read_csv('train.csv' ,
                      names=['handicappedInfants','waterProject','budgetResolution','physicianFeeFreeze','elSalvadorAid','religiousGroupsInSchools','antiSatelliteTestBan','aidToNicaraguanContras','mxMissile','immigration','synfuelsCorporationCutback','educationSpending','superfundRightToSue','crime','dutyFreeExports','exportsSouthAfrica','class']) #training set
infilePrune = pd.read_csv("prune.csv", names=['handicappedInfants','waterProject','budgetResolution','physicianFeeFreeze','elSalvadorAid','religiousGroupsInSchools','antiSatelliteTestBan','aidToNicaraguanContras','mxMissile','immigration','synfuelsCorporationCutback','educationSpending','superfundRightToSue','crime','dutyFreeExports','exportsSouthAfrica','class']) #prune set
infileTest = pd.read_csv("test.csv", names=['handicappedInfants','waterProject','budgetResolution','physicianFeeFreeze','elSalvadorAid','religiousGroupsInSchools','antiSatelliteTestBan','aidToNicaraguanContras','mxMissile','immigration','synfuelsCorporationCutback','educationSpending','superfundRightToSue','crime','dutyFreeExports','exportsSouthAfrica','class']) #testing set

infileTrain__new=pd.read_csv('train.csv' , names=['republican', 'democrat', 'class'])
infileTest_new = pd.read_csv("test.csv", names=['republican', 'democrat', 'class'])
#Just a dummy assingment
df = pd.read_csv('train.csv')
#Dummy Decision Tree, ID3 should create your own decision tree.This is a nested representation. Each node is a list. 
#if the node has one entry then it is a leaf node and performs the classification. Otherwise the node has two entries:
#the first entry is the feature ('bruises') and the second entry is a dictionary containing the tree below this node
#for example, bruises has two different possible values ('f' and 't'), therefore the dictionary has those two values
#'t' goes to a leaf node classifying 'p'. However, the 'f' branch has more decision making in this case 'gill-spacing' 
#that all lead to terminal nodes.  
dt = ['physicianFeeFreeze', {'y': ['exportsSouthAfrica', {'y': ['republican'], 
                                         'n': ['democrat'], 
                                         'noVote': ['republican']}], 
                  'n': ['democrat'],
                  'noVote': ['republican']}] 

#evaluates the 7th sample (0-index) in the training set using the dummy decision tree above 
print(evalSample(dt, df.iloc[7])) 
print(df.iloc[7])

minEntropy = float("inf") #Variable to keep track of lowest entropy found
bestFeature = "" #Variable to keep track of feature to give the lowest conditional entropy


#Your ID3 algorithm goes here! Use evalSample to evaluate your completed tree on both training and testing sets.
print(df)

def entropy(target_col):
    
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


################### 
    
###################


def InfoGain(data,split_attribute_name,target_name="class"):
    
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain
       
###################

###################


def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):
    
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    
    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.
    
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree
    
    else:
        #Set the default value for this node 
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
        
        
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters 
            subtree = ID3(sub_data,infileTrain,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)    
                
###################

###################


    
    
def predict(query,tree,default = 1):
   
    
    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
        
            result = tree[key][query[key]]
        
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result        
        

###################

###################





def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')
    

def preorder (temptree, number):
    if isinstance(temptree, dict):
        attribute = list(temptree.keys())[0]
        if temptree[attribute]['number'] == number:
            if(temptree[attribute][0]!=0 and temptree[attribute][0]!=1):
                temp_tree = temptree[attribute][0]
                if isinstance(temp_tree, dict):
                    temp_attribute = list(temp_tree.keys())[0]
                    temptree[attribute][0] = temp_tree[temp_attribute]['best_class']
            elif(temptree[attribute][1]!=0 and temptree[attribute][1]!=1):
                temp_tree = temptree[attribute][1]
                if isinstance(temp_tree, dict):
                    temp_attribute = list(temp_tree.keys())[0]      
                    temptree[attribute][1] = temp_tree[temp_attribute]['best_class']
        else:
            left = temptree[attribute][0]
            right = temptree[attribute][1]
            preorder(left, number)
            preorder(right,number )
    return temptree



def count_number_of_non_leaf_nodes(tree):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        left = tree[attribute][0]
        right = tree[attribute][1]
        return (1 + count_number_of_non_leaf_nodes(left) +  
               count_number_of_non_leaf_nodes(right)); 
    else:
        return 0;
    


def accuracy_of_the_tree(instance, tree, default=None):
    attribute = list(tree.keys())[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict): 
            return accuracy_of_the_tree(instance, result)
        else:
            return result 
    else:
        return default


def post_prune(tree):
    best_tree = tree
    for i in range(1, 5) :
        temp_tree = copy.deepcopy(best_tree)
        M = randint(1, 4);
        for j in range(1, M+1):
            n = count_number_of_non_leaf_nodes(temp_tree)
            if n> 0:
                P = randint(1,n)
            else:
                P = 0
            preorder(temp_tree, P)
        infilePrune['accuracyBeforePruning'] = infilePrune.apply(accuracy_of_the_tree, axis=1, args=(best_tree,'1') ) 
        accuracyBeforePruning = str( sum(infilePrune['class']==infilePrune['accuracyBeforePruning'] ) / (1.0*len(infilePrune.index)) )
        infilePrune['accuracy_after_pruning'] = infilePrune.apply(accuracy_of_the_tree, axis=1, args=(temp_tree,'1') ) 
        accuracy_after_pruning = str( sum(infilePrune['class']==infilePrune['accuracy_after_pruning'] ) / (1.0*len(infilePrune.index)) )
        if accuracy_after_pruning >= accuracyBeforePruning:
            best_tree = temp_tree
    return best_tree

tree = ID3(infileTrain,infileTrain,infileTrain.columns[:-1])
print(tree)
test(infileTest,tree)

tree_new=ID3(infileTrain__new,infileTrain__new,infileTrain__new.columns[:-1])
print(tree_new)
#test(infileTest_new,tree_new)

#tree2 = post_prune(tree)
