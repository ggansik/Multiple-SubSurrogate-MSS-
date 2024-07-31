
import skopt
import pyDOE
import math
import csv
from statistics import median
from sklearn.utils import check_random_state
from skopt.sampler import Lhs
import numpy as np

def initial_point_generator(space, n_samples, method):
    knob_samples = list()
    random_state = check_random_state(None).randint(0, np.iinfo(np.int32).max)
    if method == "lhs":
        # print(space)
        transformer = space.get_transformer()
        gen = Lhs()
        knob_samples = gen.generate(space.dimensions, n_samples, random_state=random_state)
        # print(knob_samples)
        space.set_transformer(transformer)
        # print(space)
    elif method == "random":
        knob_samples = space.rvs(n_samples=n_samples, random_state = random_state)
    
    return knob_samples

def load_knob_samples(BO_options):
    sampled_sets = []
    for i in range(BO_options["BO_trials"]):
        with open(str(BO_options['n_random_iter']) + "_random_set" + str(i) + ".txt") as f:
            sampled_confs = []
            csvreader = csv.reader(f)
            for line in csvreader:
                sampled_confs.append(line)
            sampled_sets.append(sampled_confs)
    return sampled_sets

def load_knob_samples_with_time(set_list):
    sampled_sets = []
    for i in set_list:
        with open("set" + str(i) + ".csv") as f:
            sampled_confs = []
            csvreader = csv.reader(f)
            for line in csvreader:
                sampled_confs.append([int(knob) for knob in line])
            sampled_sets.append(sampled_confs)
    return sampled_sets

def load_ratio_ratiofirst(filename):
    try:
        with open(filename) as f:
            csvreader = csv.reader(f)
            jobwise_ratio = []
            for line in csvreader:
                numeric_line = [int(a) for a in line]
                jobwise_ratio.append([a/sum(numeric_line) for a in numeric_line])
            
            jobwise_ratio = sorted(jobwise_ratio)
            
            median = jobwise_ratio[int(len(jobwise_ratio)/2)]
            # median = jobwise_ratio[-1]
    except:
        print("there is no file", filename)
       
    return median

def load_ratio_sumfirst(filename):
    with open(filename) as f:
        csvreader = csv.reader(f)
        jobwise_sum = [0 for i in csvreader[0]]
        for line in csvreader:
            for job, job_time in zip(jobwise_sum, line):
                job += job_time
    
        ratio_list = []
        for jobwise_sum in enumerate(jobwise_sum):
            ratio_list.append(jobwise_sum/sum(jobwise_sum))
    
    return ratio_list
        
def get_knob_samples(space, knob_name_list, num_samples, num_sample_sets, method, return_or_save, availability_check):
    
    #creation of (knob name: sampled knob value) list
    
    for set_num in range(num_sample_sets):
        random_knob_dict_list = []
        random_knob_value_list = []
        if method == "random":
            for i in range(num_samples):
                
                random_knob_dict = {}
                random_knobs = None
                if availability_check:
                    while True:
                        random_knobs = space.rvs(1)
                        for knob_name, knob_value in zip(knob_name_list, random_knobs[0]):
                            random_knob_dict[knob_name] = knob_value
                        
                        if resource_availability_check(random_knob_dict):
                            random_knob_dict_list.append(random_knob_dict)
                            random_knob_value_list.append(random_knobs)
                            break
                else:
                    random_knobs = space.rvs(1)
                    for knob_name, knob_value in zip(knob_name_list, random_knobs[0]):
                        random_knob_dict[knob_name] = knob_value
                    random_knob_dict_list.append(random_knob_dict)
                    random_knob_value_list.append(random_knobs)
                    
        elif method == "LHS" or method == "lhs":
            print("lhs started")
            while True:
                random_knob_dict_list = []
                random_knob_value_list = []
                lhd = pyDOE.lhs(len(knob_name_list), samples=num_samples)
                #scale to original domain
                for sample in lhd.tolist():
                    
                    random_knob_dict = {}
                    random_knobs = []
                    for bound, value in zip(space.bounds, sample):
                        random_knobs.append(int(round((bound[1] - bound[0]) * value + bound[0])))
                    # print(random_knobs)
                        
                    for knob_name, knob_value in zip(knob_name_list, random_knobs):
                        random_knob_dict[knob_name] = knob_value
                        
                    # print(random_knob_dict)
                    if resource_availability_check(random_knob_dict):
                        # print("hello")
                        random_knob_dict_list.append(random_knob_dict)
                        random_knob_value_list.append(random_knobs)
                    else:
                        break
                
                if len(random_knob_dict_list) == num_samples:
                    
                    print(random_knob_value_list)
                    break
                else:
                    print(len(random_knob_dict_list))
        
        if return_or_save == "return":
            return random_knob_dict_list, random_knob_value_list
        elif return_or_save == "save":
            with open(str(num_samples) + "_" + method + "_set" + str(set_num) + ".txt", "w") as f:
                csvwriter = csv.writer(f)
                for conf in random_knob_value_list:
                    csvwriter.writerow(conf[0])
            f.close()
        



def resource_availability_check(a_knob_sample):
    # executor_cores, spark.driver.cores, executor_memory, driver_memory: maximum capacity of the container specified yarn-container?
    # memoryOverhead, offheap size: 0 to max memory capacity of yarn container
    # sum of executor memory + executor_ memoryoverhead + spark.memory_offheap.size -> smaller than the memory capacity of the yarn container

    # executor.instances x resource amount of a single process -> less than the total amount of resources
    if a_knob_sample['spark.executor.instances'] * a_knob_sample['spark.executor.cores'] >= 80: #total vcores in the cluster
        return False
    
    if a_knob_sample['spark.executor.instances'] * a_knob_sample['spark.executor.memory'] >= 200: #total memory in the cluster
        return False
    
    if a_knob_sample['spark.executor.memory'] + a_knob_sample['spark.executor.memoryOverhead'] * 0.384 >= 20: #maximum memory in a single container(executor)
        return False
    

    return True

if __name__ == '__main__':
    
    BO_options = {"kernel": "matern", # matern+constant, RBF+constant
                "max_iter": 100, # total iteration for BO
                "n_random_iter": 10, # random initialization for BO
                "sampler": "random", # sampling strategy for random initialization
                "acquisition_func": "AEI",
                "n_random_samples": 100, # the number of random samples for acquisition function (minimize, L-BFGS)
                "termination_EI": 10, # termination condition before max_iter
                "BO_trials": 10, # for the mitigation of the randomness of BO
                "target_app_time": 41000}
    
    load_knob_samples(BO_options)