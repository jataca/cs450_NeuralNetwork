from __future__ import division
from IrisList import *
from Weights import weights, newWeights, layers
import math

learningRate = 0.1 #neural networks need a constant to decide how "fast" the network should learn.
lastLayer = len(layers) - 1
one = []
one.append(1)
epocs = 50
for epoc in range (epocs):
    for instances in range(len(trainingSetData)): #for instances in range(len(trainingSetData)):
        # inputs at the start layer
        inputs = [[]]
        activationOfNode = [[]]
        errorOfNode = [[]]
        for i in range(len(trainingSetData[0])): # feed forward
            inputs[0].append(trainingSetData[instances][i])
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

        #calculate error of targets (Error of Output Node = activation value (1 - activation value)(activation value - target value)
        errorOfNode[lastLayer][trainingSetTargets[instances]] = 1 # set error of real value to one
        for i in range(len(errorOfNode[lastLayer])):
            activationValue = activationOfNode[lastLayer][i]
            errorOfNode[lastLayer][i] = activationValue * (1 - activationValue) * (activationValue - errorOfNode[lastLayer][i])

        #calculate weight on outer layer
        currentWeight = weightCounter - 1
        lastHiddenInputs = len(inputs) - 2
        currentNode = len(inputs[lastHiddenInputs]) - 1
        #print "inputs", inputs
        while currentNode >= 0:
            currentOutputNode = len(inputs[len(inputs) - 1]) - 1
            while currentOutputNode >= 0:
                #print "input", inputs[lastHiddenInputs][currentNode]
               # print "OuterNode", inputs[len(inputs) - 1][currentOutputNode]
                #print "currentWeight", weights[currentWeight]
                newWeights[currentWeight] = weights[currentWeight] - learningRate  *  errorOfNode[len(errorOfNode) - 2][currentOutputNode]*inputs[lastHiddenInputs][currentNode]
                currentWeight -= 1
                currentOutputNode -= 1
            currentNode -=1

        #calculate error of hidden nodes
        targetWeight = weightCounter - 1
        #print layers
        hiddenLayers = len(layers) - 2 # number of hidden layers
        while hiddenLayers >= 0: # go through all hiddenlayers
            currentNode = len(inputs[hiddenLayers + 1]) - 1
            isBias = True
            #print "hiddenLayer", hiddenLayers
            while currentNode >= 0:
                errorSum = 0
                #print "currentNode", currentNode
                kNode = len(activationOfNode[hiddenLayers + 1]) - 1
                while kNode >= 0:
                    #print "kNode", kNode
                    if isBias == False:
                        #print "weight", weights[currentWeight]
                        #print errorOfNode[hiddenLayers][currentNode] # the error that needs to be updated
                        errorSum += weights[currentWeight] * errorOfNode[hiddenLayers + 1][kNode]
                    targetWeight -= 1
                    kNode -=1
                    if isBias == False:
                        activationValue = activationOfNode[hiddenLayers][currentNode] * (1 - activationOfNode[hiddenLayers][currentNode])
                        errorOfNode[hiddenLayers][currentNode] = activationValue * errorSum
                isBias = False
                currentNode -= 1
            hiddenLayers -= 1
        #print "error of nodes", errorOfNode
        #print "inputs", inputs
        #print " activationOfNodes", activationOfNode
        #print "instance", instances, weights

        #calculate weights in hidden layers
        hiddenInputLayer = len(inputs) - 3
        while hiddenInputLayer >= 0: # go through all layers
            currentInput = len(inputs[hiddenInputLayer]) - 1
           # print "input level",  hiddenInputLayer
            while currentInput >= 0:
                currentNode = len(activationOfNode[hiddenInputLayer]) - 1
                #print "input number", currentInput
                while currentNode >= 0:
                    #print "current node", currentNode
                    #recalculate weight
                    newWeights[currentWeight] = weights[currentWeight] - learningRate * inputs[hiddenInputLayer][currentInput] * errorOfNode[hiddenInputLayer][currentNode]
                    #print newWeights[currentWeight]
                    if newWeights[currentWeight] == 0:
                        newWeights[currentWeight] = 0.5
                    currentNode -= 1
                    currentWeight -= 1
                currentInput -= 1
            hiddenInputLayer -= 1

        for i in range (len(weights)):
            tempWeight = newWeights[i]
            weights[i] = tempWeight

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

