import numpy as np
import matplotlib.pyplot as plt
import random
import time


def parseTextFile():
    """Function to parse the 'BankProblem.txt' text file"""

    weightList = []
    valueList = []

    # Open 'BankProblem.txt' text file to get the security van capacity, each bag's weight and its value
    # Read each line from the file and append the values in the variables
    with open("BankProblem.txt", "r") as file:
        for line in file:
            if 'weight:' in line:
                weightList.append(float(line.split(':')[1].strip()))
            elif 'value:' in line:
                valueList.append(float(line.split(':')[1].strip()))
            elif 'security van capacity:' in line:
                securityVanCapacity = float(line.split(':')[1].strip())

    return securityVanCapacity, valueList, weightList


#def printMatrix(matrix: list[list[int]]) -> None:
#    """Function for printing the matrix"""
#
#    for row in matrix:
#        print(row)
#
#    return


def getCeilValue(arr: list, val: float) -> int:
    """Function to get the ceil value for any probability val"""

    left = 0
    right = len(arr)

    while left < right :     
        mid = left + ((right - left) >> 1)
        if val > arr[mid] : 
            left = mid + 1
        else : 
            right = mid 
      
    if arr[left] >= val : 
        return left
    else : 
        return -1


def updateProbability(probList: list, pheromoneList: list, alpha: int, heuristicList: list, beta: int) -> list:
    """Function to update the probability list according the pheromone and heuristic"""

    probList = np.power(pheromoneList, alpha) * np.power(heuristicList, beta)
    probList = probList / np.sum(probList)
    return probList


def getCummulativeProbability(probList: list) -> list:
    """Function to get the cummulative probability"""

    cummulativeProbList = [0.0]
    
    for prob in probList:
        cummulativeProbList.append(cummulativeProbList[-1] + prob)

    cummulativeProbList.pop(0)
    return cummulativeProbList


# Main function
def main():

    startTime = time.time()

    # Parameters to be changed
    numberOfIterations = 10000
    numberOfAnts = 500

    # Fixed parameters for calculating probability
    alpha = 1
    beta = 2
    rho = 0.8

    # Store the values from the text file
    securityVanCapacity, valueList, weightList = parseTextFile()
    
    print("Security van capacity:", securityVanCapacity)
    print("Weight list size:", len(weightList))
    print("Value list size:", len(valueList))

    # Number of cities or weights
    numOfCities = len(weightList)

    # Distance matrix
    distanceMatrix = [[0]*numOfCities for _ in range(numOfCities)]

    # Construct the distanceMatrix from weightList
    for r in range(numOfCities):
        for c in range(numOfCities):
            if r <= c:
                distanceMatrix[r][c] = weightList[c] / valueList[c]

    for r in range(numOfCities):
        for c in range(numOfCities):
            distanceMatrix[c][r] = distanceMatrix[r][c]

    # Pheromone matrix, initialise with 1
    pheromoneList = [1]*numOfCities

    citiesInAntTour = []
    chosenWeightsList = []
    currMax = 0.0
    currWeight = 0.0
    currVal = 0.0

    maxValuesOfAllIters = []

    # Main loop start
    for _ in range(int(numberOfIterations/numberOfAnts)):

        # Pheromone list
        pheromoneList = [1]*numOfCities
        currMax = 0.0

        # Loop for the ants
        for _ in range(numberOfAnts):
            
            # Heuristic matrix
            heuristicList = [round(valueList[i] / weightList[i], 4) for i in range(numOfCities)]
            probList = np.divide(heuristicList, np.sum(heuristicList)).tolist()
            cummulativeProbList = getCummulativeProbability(probList)

            # Clear the values initially
            citiesInAntTour.clear()
            chosenWeightsList.clear()
            currWeight = 0.0
            currVal = 0.0

            # Loop to fill van until full
            while currWeight < securityVanCapacity:
                
                # Get random value from 0 to 1
                randomVal = random.uniform(0, 1)
                val = getCeilValue(cummulativeProbList, randomVal)
                
                if currWeight + weightList[val] <= securityVanCapacity:
                    heuristicList[val] = 0
                    currWeight = currWeight + weightList[val]
                    currVal = currVal + valueList[val]
                    citiesInAntTour.append(val)
                    chosenWeightsList.append(weightList[val])
                else:
                    break

                # Update transition probability to another city
                probList = updateProbability(probList, pheromoneList, alpha, heuristicList, beta)
                cummulativeProbList = getCummulativeProbability(probList)
            
            # Update pheromone after ant traversal
            pheromoneList = np.multiply(pheromoneList, 1-rho)
            for city in citiesInAntTour:
                pheromoneList[city] = pheromoneList[city] + (1/np.sum(chosenWeightsList))

            # Get the maximum value
            currMax = max(currMax, currVal)
            maxValuesOfAllIters.append(currMax)

    endTime = time.time()
    print("Time taken to execute:", endTime - startTime)

    print("Maximum value after all the interations:", np.max(maxValuesOfAllIters))
    
    # Plot the graph
    plt.title("Ant Colony Optimization")
    plt.plot(range(numberOfIterations), maxValuesOfAllIters)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Total value of weights in van (Pounds)")
    plt.savefig("alpha"+str(alpha)+"_"+"beta"+str(beta)+"_"+"rho"+str(rho)+"_"+"ants"+str(numberOfAnts)+".png")
    plt.show()  

    with open("Results.txt", 'a') as file:
        file.write("Number of iterations: "+ str(numberOfIterations)+"\n")
        file.write("Number of ants: "+ str(numberOfAnts)+"\n")
        file.write("Alpha: "+ str(alpha)+"\n")
        file.write("Beta: "+ str(beta)+"\n")
        file.write("Rho: "+ str(rho)+"\n")
        file.write("Maximum value after all the iterations: "+ str(np.max(maxValuesOfAllIters))+"\n")

    return


if __name__ == "__main__":
    main()