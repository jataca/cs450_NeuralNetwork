#import iris and random
from sklearn import datasets
iris = datasets.load_iris()
import random

# create a new array that will store the iris data and targets
bigArray = []

amountPerTarget = (len(iris.data) / len(iris.target_names))

#Fill bigArray with iris data and target.
for i, iris.data in enumerate(iris.data):
    tempTarget = i/amountPerTarget
    tempArray = [iris.data, tempTarget]
    bigArray.append(tempArray)

#Randomize iris
random.shuffle(bigArray)

#Set training and test set
trainingSet = bigArray[:105] #70%
testSet = bigArray[105:]  #30

#make list of targets
trainingSetTargets = []
for i in range(len(trainingSet)):
    trainingSetTargets.append(trainingSet[i][1])

# trianing set data list
trainingSetData = []
for i in range(len(trainingSet)):
    trainingSetData.append(trainingSet[i][0])

#list of test targets
testSetTargets = []
for i in range(len(testSet)):
    testSetTargets.append(testSet[i][1])

# list of test set data
testSetData = []
for i in range(len(testSet)):
    testSetData.append(testSet[i][0])
