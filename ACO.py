import numpy as np
import matplotlib.pyplot as plt
import random


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


def getCeilValue(arr: list, valueToSearch: float) -> int:
    """Function to get the ceil value for any probability val"""

    left = 0
    right = len(arr)

    # Use binary search to search for the value
    while left < right :     
        mid = left + ((right - left) >> 1)
        if valueToSearch > arr[mid] : 
            left = mid + 1
        else : 
            right = mid 
    
    # Return the value just above the value
    if arr[left] >= valueToSearch : 
        return left
    else : 
        return -1


def updateProbability(probList: list, pheromoneList: list, alpha: int, heuristicList: list, beta: int) -> list:
    """Function to update the probability list according the pheromone and heuristic"""

    # Update the probability based on the pheromone and heuristic list, using alpha and beta parameters
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

    print("Ant Colony Optimisation")

    # Total number of fitness evaluations
    totalFitnessIterations = 10000
    
    # Parameters to be tested via experiments
    populationSize = 50 # population size, p
    m = 1.0 # amount of pheromone deposited according to fitness
    rho = 0.7 # evaporation rate, e

    # Fixed parameters for calculating probability
    alpha = 2
    beta = 4

    # Store the values from the text file
    securityVanCapacity, valueList, weightList = parseTextFile()

    # Number of cities or weights
    numOfCities = len(weightList)

    # Pheromone matrix, initialise with 1
    pheromoneList = [1]*numOfCities

    # Index of cities in the tour
    citiesInAntTour = []
    currentMax = 0.0
    currentWeight = 0.0
    currentVal = 0.0

    # Cummulative value of the bag in one iteration
    cummulativeValueList = []

    # Cummulative values of the bag in all iterations
    cummulativeValueMatrix = []

    # Max values of iterations
    maxValuesOfAllIters = []

    # Iteration count
    iterCount = 1

    # Main loop start
    for _ in range(int(totalFitnessIterations/populationSize)):

        # Pheromone list
        pheromoneList = [1]*numOfCities
        currentMax = 0.0

        # Loop for the ants
        for _ in range(populationSize):
            
            # Heuristic matrix
            heuristicList = [round(valueList[i] / weightList[i], 4) for i in range(numOfCities)]
            probList = np.divide(heuristicList, np.sum(heuristicList)).tolist()
            cummulativeProbList = getCummulativeProbability(probList)

            # Clear the values initially
            citiesInAntTour.clear()
            currentWeight = 0.0
            currentVal = 0.0
            cummulativeValueList.clear()

            # Loop to fill van until full
            while currentWeight < securityVanCapacity:
                
                # Get random value from 0 to 1
                randomVal = random.uniform(0, 1)
                val = getCeilValue(cummulativeProbList, randomVal)
                
                # Check if the weight can be put in the security van
                # If yes, then choose the weight and update the heuristic value of it to 0
                # Update the current value of weights and valuables
                if currentWeight + weightList[val] <= securityVanCapacity:
                    heuristicList[val] = 0
                    currentWeight = currentWeight + weightList[val]
                    currentVal = currentVal + valueList[val]
                    citiesInAntTour.append(val)
                else:
                    break

                # Update transition probability to another city
                probList = updateProbability(probList, pheromoneList, alpha, heuristicList, beta)
                cummulativeProbList = getCummulativeProbability(probList)

                cummulativeValueList.append(currentVal)
        
            cummulativeValueMatrix.append(cummulativeValueList[:])

            # Update pheromone after ant traversal
            pheromoneList = np.multiply(pheromoneList, 1-rho)
            for city in citiesInAntTour:
                pheromoneList[city] = pheromoneList[city] + (m * currentVal /currentWeight)

            # Get the maximum value
            currentMax = max(currentMax, currentVal)
            maxValuesOfAllIters.append(currentMax)
            iterCount = iterCount+1


    print("Maximum value after all the interations:", np.max(maxValuesOfAllIters))
    
    # Get maximum values after all the iterations
    maxValuesAfterAllIterations = []
    for i in range(len(cummulativeValueMatrix)):
        maxValuesAfterAllIterations.append(cummulativeValueMatrix[i][-1])

    print("Average value after all iterations:", np.average(maxValuesAfterAllIterations))

    # Uncomment to plot graph
    # Plot the graph
    #plt.title("Ant Colony Optimization")
    #plt.plot(range(len(maxValuesAfterAllIterations)), maxValuesAfterAllIterations, marker='o')
    #plt.xlabel("Number of Iterations",fontsize=18)
    #plt.ylabel("Maximum value of weights (Pounds)",fontsize=18)
    #plt.grid()
    #plt.show()

    return


if __name__ == "__main__":
    main()