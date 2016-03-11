from IrisData import *
import math

layers = []
layers.append(3) # must have at least 1 hidden layer
layers.append(3)

#set of possible targets and add to layers
setOfTargets = set(trainingSetTargets)
listOfUniqueTargets = list(setOfTargets)
layers.append(len(listOfUniqueTargets))

# inputs at the start layer
inputs = [[]]
for i in range(len(trainingSetData[0])):
    inputs[0].append(trainingSetData[0][i])

weights = [] # to store original weights
newWeights = [] # to store new weights as we to the feed back
for i in range(len(layers)):
    nextI = i + 1
    answers = []
    inputs[i].append(-1) # add bias node to input layer
    for j in range(layers[i]):
        answer = 0.0
        for k in range(len(inputs[i])):
            answer += (inputs[i][k] * .5)
            weight = random.uniform(-1,  1)
            weights.append(weight)
            newWeights.append(0)
        answer = 1/(1 + pow(math.e, answer))
        answers.append(answer)
    inputs.append([])
    for k in range(len(answers)): #set answers to be inputs of next layer
        inputs[nextI].append(answers[k])


