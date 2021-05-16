# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 03:55:44 2021

@author: Samer Kazem Qarajai / Student ID: 20107283
"""

#from deap import base, creator
import random
from deap import tools
import numpy as np


#from scipy.stats import levy_stable as levys
from scipy.special import gamma
from math import sin, pi

from deap import base, creator
import random

import matplotlib.pyplot as plt

import numpy as np
import random
from math import sqrt
import random as ra
import pickle

def read_data_to_train_test(fileName, delimiter = ' ', train_size = 0.9, zero_index = True):
    test_data = None
    tmp_lst = []
    matrix_size = None
    
    with open(fileName, 'r') as f:
        line = f.readline().strip().split(delimiter)
        matrix_size = (int(line[0]), int(line[1]))
        if zero_index:
            for line in f:
                line = line.strip().split(delimiter) 
                tmp_lst.append( [int(line[0]), int(line[1]), float(line[2])] )
        else:
            for line in f:
                line = line.strip().split(delimiter) 
                tmp_lst.append( [int(line[0])-1, int(line[1])-1, float(line[2])] )
            
    random.shuffle(tmp_lst)
    cut_point = int(len(tmp_lst)*train_size)
    test_data, training_data = tmp_lst[cut_point:], tmp_lst[:cut_point]
    
    return training_data, test_data, matrix_size
    
def write_data(fileName, delimiter = ' ', matrixSize = None, data = None):
    with open(fileName, 'w') as f:
        f.write(str(matrixSize[0])+delimiter+str(matrixSize[0])+'\n')
        str_lst = []
        for d in data:
            str_lst.append(delimiter.join([str(i) for i in d]))
        f.write('\n'.join(str_lst))
        
            
def rmse(real_mat, pred_mat, n):
    return np.linalg.norm(real_mat-pred_mat) * sqrt(1./n)
   
def create_matrix(edgeList, size):
    mat = np.zeros(size)
    
    for i in edgeList:
        mat[i[0]][i[1]] = i[2]
    return mat
          



# statistics registeration
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
#stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

def run_ga(new_inds_ratio = 0.1, CXPB = 0.9, MUTPB = 0.2, LSPB = 0.2, NGEN = 100, ind_type = np.ndarray,
           ind_size = None, pop_size = 50, ind_gen = None, mate = None,
           mutate = None, select = tools.selTournament, evaluate = None, local_search = None, curve_label = "GA"):
           
    toolbox = base.Toolbox()
           
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", ind_type, fitness=creator.FitnessMin)
    
    
    toolbox.register("attribute", ind_gen)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    pop = toolbox.population(n=pop_size)
    
    toolbox.register("mate", mate) #tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=0.1)#tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("local_search", local_search)
    toolbox.register("select", select, tournsize= 5)#len(pop)/10)
    toolbox.register("evaluate", evaluate)

    #list(map(int,pay))
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    min_fit_lst = []
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, int(pop_size*(1-new_inds_ratio)))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        for indvidual in offspring:
            if random.random() < LSPB:
                toolbox.local_search(indvidual)
                del indvidual.fitness.values

        # Generate new random individuals
        offspring += toolbox.population(n=(pop_size-len(offspring)))
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # printing statistics
        record = stats.compile(pop)
        min_fit = record['min']
        max_fit = record['max']
        std = record['std']
        min_fit_lst.append(min_fit)
        
        print ("gen #%d: stats min:%f max:%f std:%f" %(g, min_fit, max_fit, std))
    #plt.plot(min_fit_lst, label=curve_label)
    #plt.legend()
    #plt.show()
    return (pop,min_fit_lst)


dataset = ('movielens', 'D:\BCU University\Course\CMP7213-A-S2-20201 - Dr Atif Azad\Assessment\Assessment 1.2\GA/final_set.csv')
#dataset = ('movelens 1m', '../resources/ml-1m/ratings.dat')
train, test, mSize = read_data_to_train_test(dataset[1], zero_index = False)

V = create_matrix(train, mSize)
maskV = np.sign(V)

r_dim = 10
eps = 1e-10

def generate_ind():
    r = np.random.rand(r_dim)
    #r = np.random.normal(scale=1./r_dim, size = r_dim)
    #r = np.maximum(r, eps)
    return r
      
def evaluate_ind(ind):
    W, H = ind[:mSize[0]], ind[mSize[0]:]
    predV = maskV * W.dot(H.T)
    fit = rmse(V, predV, len(train))#np.linalg.norm(V-predV)
    
    #if np.min(ind)<0:
    #    fit *= 100
    return fit,
  

def linear_combinaiton_CX(ind1, ind2):
    rand1, rand2= random.random(), random.random()
    rand1_c, rand2_c = 1-rand1, 1-rand2
    
    ind1[:], ind2[:] = (ind1.copy()*rand1 + ind2.copy()*rand1_c), (ind1.copy()*rand2 + ind2.copy()*rand2_c)
    return ind1, ind2
    
def mMut(ind, indpb):
    mu=0
    sigma=1
    tools.mutGaussian(ind, mu, sigma, indpb)
    ind = np.maximum(ind, eps)
    return ind

def mixMut(ind, indpb):
    if random.random() < 0.5:
        return levyMut(ind, indpb)
    return mMut(ind, indpb)

##############################################
sigma = None
def mantegna_levy_step(beta=1.5, size=1):
    global sigma
    if sigma == None:
        sigma = gamma(1+beta) * sin(pi*beta/2.)
        sigma /= ( beta * gamma((1+beta)/2) * pow(2, (beta-1)/2.) )
        sigma = pow(sigma , 1./beta)
    u = np.random.normal(scale=sigma, size=size)
    v = np.absolute(np.random.normal(size=size))
    step = u/np.power(v, 1./beta)
    
    return step
###############################################
    
def levyMut(ind_, indpb=0.1):
    ind = ind_.copy()
    steps = mantegna_levy_step(size=(ind.shape)) 
    
    levy = 1.5*gamma(1.5)*sin(pi*1.5/2)/(pi*np.power(steps, 2))
    
    ind += 0.2 * levy
    
    if evaluate_ind(ind) < evaluate_ind(ind_):
        ind_[:] = ind[:]
    
    return ind


def print_results( predMat = None, nFeatures=None, train_data=None, test_data=None, method_name=None, 
                    nIterations=None, dataset_name=None, method_details=[]):
    results ='\n############# results of %s method: \n'%(method_name) 
    results += '## dataset: %s \n' %(dataset_name)
    results += '## number of latent features: %d \n'%(nFeatures)
    results += '## number of training iteratins: %d \n'%(nIterations)
    
    for d in method_details:
        results += '## %s: %s \n' %(str(d[0]), str(d[1]))
    
    trainMat = create_matrix(train_data, predMat.shape)
    testMat = create_matrix(test_data, predMat.shape)
    
    #calculating rmse
    predMat = np.maximum(predMat, 1)
    predMat = np.minimum(predMat, 5)
    train_rmse = rmse(trainMat, predMat*np.sign(trainMat), len(train_data))
    test_rmse = rmse(testMat, predMat*np.sign(testMat), len(test_data))
    
    results += '## training RMSE: %f \n'%(train_rmse)
    results += '## testing RMSE: %f \n'%(test_rmse)
    
    results +='########################################################'
    
    with open('../results.txt', 'a') as f:
        f.writelines(results)
    
    print (results)
    return results

def Average_Fitness(lst):
    return sum(lst) / len(lst)

       
if __name__ == '__main__':
    
    avg_fit = []
    best_sol = 0 # just a random value the
    best_sol_fit = 900 # if it is a  if you are getting the max
    GA_Run_Number = 0
    Best_pop=0
    for i in range(0,30):
        ra.seed(i)
        
        pop_size = 100
        mate = linear_combinaiton_CX
        mutate = mMut
        MUTPB = 0.1
        local_search = levyMut
        CXPB = 0.9
        LSPB = 0.9
        new_inds_ratio = 0.25
        NGEN = 30
        method_name = 'GA_LS-beta=1'
      
        pop = run_ga(ind_size = mSize[0]+mSize[1], pop_size = pop_size, mate = mate, mutate = mutate, MUTPB = MUTPB, 
                        evaluate = evaluate_ind, local_search = local_search, CXPB = CXPB, LSPB = LSPB,
                        ind_gen = generate_ind, new_inds_ratio = new_inds_ratio, NGEN = NGEN, curve_label = method_name)
       
        #printng results:
        minInd = min(pop[0] , key = lambda ind: ind.fitness.values[0])
        avg_fit.append(minInd.fitness.values[0])
      
        if best_sol_fit > minInd.fitness.values[0]:
            best_sol_fit = minInd.fitness.values[0]
            best_sol = minInd#the new  solution
            Best_pop=pop
            GA_Run_Number=i
        
        
    W, H = best_sol[:mSize[0]], minInd[mSize[0]:].T
    predicted_matrix=W.dot(H)
    ga_results = print_results(predicted_matrix, r_dim, 
                               train_data = train, test_data = test, 
                               method_name = method_name, nIterations = NGEN, 
                               dataset_name = dataset[0], 
                               method_details = [('pop_size',pop_size),
                                    ('crossover', mate.__name__),
                                    ('crossover prob', CXPB),
                                    ('mutation',mutate.__name__),
                                    ('mutation prob', MUTPB),
                                    ('local search', local_search.__name__),
                                    ('local search prob', LSPB),
                                    ('new random individuals ratio', new_inds_ratio),
                                    ('GA Run', GA_Run_Number),
                                    ('Best Solution Fintness Value', minInd.fitness.values[0])]
                               )
    

    
    Second_GA_Run_Best_pop=Best_pop
    
     # open a pickle file
    filename = 'GA_v2_pickle.pk'
    
    with open(filename, 'wb') as fi:
        # dump your data into the file
        pickle.dump(Second_GA_Run_Best_pop[1], fi)

    
    Avg_Fitness = Average_Fitness(avg_fit)    
    plt.plot(Best_pop[1], label='Best GA')
    plt.legend()
    plt.show()
    