from nodes_generator import NodeGenerator
from simulated_annealing import SimulatedAnnealing


def main():
    '''set the simulated annealing algorithm params'''
    temp = 1000
    final_temper = 0.00000001
    alpha = 0.9995
    number_of_iter = 100000

    '''set the dimensions of the grid'''
    size_width = 200
    size_height = 200

    '''set the number of nodes'''
    population_size = int(input("Enter number of nodes (0 - 15)): "))

    '''generate random list of nodes'''
    nodes = NodeGenerator(size_width, size_height, population_size).generate()

    '''run simulated annealing algorithm with 2-opt'''
    sa = SimulatedAnnealing(nodes, temp, alpha, final_temper, number_of_iter)
    sa.anneal()

    '''animate'''
    sa.animateSolutions()

    '''show the improvement over time'''
    sa.plotLearning()


if __name__ == "__main__":
    main()
