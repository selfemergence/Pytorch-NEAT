# -*- coding: utf-8 -*-

from population import Population
from parameter import params
#import cv2

#import random
#random.seed(0)
import numpy as np
np.random.seed(1111)

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]
#xor_inputs = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), \
#              (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), (1, 1, 1, 1)]
#xor_outputs = [   (0,),     (0,),     (0,),     (0.,),\
#               (1,),     (1,),     (1,),     (1.,)]

'''import activation
params['hidden_act_fn'] = activation.relu
params['output_act_fn'] = activation.sigmoid'''

params['NODE_MUTATE_PROB'] = 0.03 # 0.03
params['CONN_MUTATE_PROB'] = 0.05
params['pop_size'] = 50
params['gens'] = 50

params['INPUTS'] = 2
params['OUTPUTS'] = 1

params['epochs'] = 20
      
def fitness(pop):
    """
    Recieves a list of pop. Modify ONLY their
    fitness values
    """
    for agent in pop.members:
        agent.fitness = 4
#        nw_activate = agent.network.compile_network()
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = agent.network.getOutput(xi)
            agent.fitness -= (output[0] - xo[0]) ** 2
                             
def fitness_learning(pop):
    """
    Recieves a list of pop. Let each agent go
    through lifetime learning, and compute fitness at the end.
    """
    for agent in pop.members:
        agent.fitness = 4
        for xi, xo in zip(xor_inputs, xor_outputs):
            # lifetime learning --> fitness
            #epochs = np.random.randint(0,params['epochs'])
            epochs = params['epochs']
            for epoch in range(epochs):
                output = agent.network.getOutput(xi)
                error_vector = (xo[0]-output[0],)
                agent.network.backpropagate(error_vector)
                                
            output = agent.network.getOutput(xi)
            agent.fitness -= (output[0] - xo[0])**2
            
if __name__ == "__main__":
    pop = Population(pop_size=params['pop_size'], n_inputs=2, n_outputs=1)
    fitness_learning(pop)
    #fitness(pop)
    
    avg_fitness = []
    best_fitness = []
    species = []
    innovation = []
    
    for gen in range(params['gens']):
        print(">Gen ", gen, " Species: ", len(pop.species))
        species.append(len(pop.species))
        
        best = max(pop.members)
        best_fitness.append(best.fitness)
        print(" Best fitness ", best.fitness, " Conn: ", \
              len(best.genome.connections), ", Neurons: ", len(best.genome.neurons))
        avg_fitness.append(pop.compute_average_fitness())
        print("\t Average Fitness ", pop.average_fitness)
        '''for conn in best.genome.connections.values():
            print("\t", conn)
        
        for neuron in best.genome.neurons.values():
            print("\t", neuron)'''
            
        #Draw best Network
#        cv2.imshow("Network", best.network.draw(1000, 800))
#        cv2.waitKey(2)
            
        print("\t innovation number ", params['innov_no'])
        innovation.append(params['innov_no'])
        
        pop.evolve()
        
        fitness_learning(pop)
        #fitness(pop)
        
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(best_fitness)
plt.plot(avg_fitness)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(['Best', 'Avg'], bbox_to_anchor=(0., 1.02, 1., .102), \
           loc=3, borderaxespad=0., ncol=3, mode="expand")
plt.savefig('result/fitness.png')
plt.show()

plt.figure(2)
plt.plot(species)
plt.xlabel("Generation")
plt.ylabel("Species")
plt.savefig('result/species.png')
plt.show()

plt.figure(3)
plt.plot(innovation)
plt.xlabel("Generation")
plt.ylabel("Innovation")
plt.savefig('result/innovation.png')
plt.show()



        