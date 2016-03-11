from __future__ import division
from TrainNetwork import *


print "new weights", weights

correct = 0
for instances in range(len(testSetData)): #for instances in range(len(trainingSetData)):
    # inputs at the start layer
    inputs = [[]]
    activationOfNode = [[]]
    errorOfNode = [[]]
    for i in range(len(testSetData[0])): # feed forward
        inputs[0].append(testSetData[instances][i])
    weightCounter = 0
    for i in range(len(layers)):
        nextI = i + 1
        activationOfNode.append([])
        errorOfNode.append([])
        inputs[i].append(-1) # add bias node to input layer
        for j in range(layers[i]): # go through each node
            answer = 0.0
            for k in range(len(inputs[i])): #go through each input
                answer += (inputs[i][k] * weights[weightCounter])
                weightCounter += 1
            answer = 1/(1 + pow(math.e, answer))
            activationOfNode[i].append(answer)
            errorOfNode[i].append(0)
        inputs.append([])
        for k in range(len(activationOfNode[i])): #set answers to be inputs of next layer
            inputs[nextI].append(activationOfNode[i][k])


    guess = 0
    #print "-----"
    #print instances
    #print testSetTargets[instances]
    #print inputs[len(layers)]
    #print inputs
    #print len(inputs[len(layers)])
    #print inputs[len(layers)][guess]

    for i in range(len(inputs[len(layers)])):
        if inputs[len(layers)][i] > inputs[len(layers)][guess]:
            guess = i
    #print guess
    if guess == testSetTargets[instances]:
        correct += 1

print "percentage", correct/instances

