import pandas as pd
import math

#Evaluates a sample using the given decision tree dt. Returns true if the decision tree gets it right, false if not
def evalSample(dt, sample):   
    while (len(dt) == 2):
        dt = dt[1][sample[dt[0]]]
    return (dt[0] == sample['party'])

infileTrain = pd.read_csv("train.csv") #training set
infilePrune = pd.read_csv("prune.csv") #prune set
infileTest = pd.read_csv("test.csv") #testing set

#Just a dummy assingment
df = infileTrain 
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