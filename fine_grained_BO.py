import subprocess
import skopt
from skopt.space.space import Integer,Categorical,Real
from skopt.utils import normalize_dimensions
import sys

from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
from skopt.sampler import InitialPointGenerator, Lhs
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.utils import check_random_state
from sklearn.base import clone
import numpy as np
import csv
import warnings
import multiprocessing

from scipy.stats import norm
from scipy.optimize import minimize, fmin_l_bfgs_b
import os
import time

sys.path.append("../utils")
from log_analyzer import log_analyzer
from sparksubmit import sparksubmit_command, run
from knobs import initial_point_generator, load_ratio_ratiofirst
from make_BO_init_data import make_BO_init_data
from bayesian_optimization import suggest, new_AEI, new_AAEI


def rm_output_path(appname):
    #e.g., Wordcount_small_.jar
    workload = appname.split('_')[0]
    datasize = appname.split('_')[1]
    try:
        os.system("hdfs dfs -rm -R /user/root/HiBench/" + workload + "/" + datasize + "/Output")
    except:
        print("there is no file")

def make_iter1_to_real(reduced_app, BO_options):
    real_iter1 = list()
    for ratio in reduced_app[:BO_options['reduced_iteration_interval'][0]+2]:
        real_iter1.append(ratio)


        # for it in range(BO_options['reduced_iteration_interval'][1]-BO_options['reduced_iteration_interval'][0]+1): #num_iteration
        #     reduced_ratio.append(xi_distribute_ratio[BO_options['iteration_interval'][0]+1])


    for ratio in reduced_app[BO_options['reduced_iteration_interval'][1]+2:]:
        real_iter1.append(ratio)

    return real_iter1


def make_reduced_app(original_app, BO_options):
    reduced_app = list()
    for ratio in original_app[:BO_options['iteration_interval'][0]+2]:
        reduced_app.append(ratio)


        # for it in range(BO_options['reduced_iteration_interval'][1]-BO_options['reduced_iteration_interval'][0]+1): #num_iteration
        #     reduced_ratio.append(xi_distribute_ratio[BO_options['iteration_interval'][0]+1])


    for ratio in original_app[BO_options['iteration_interval'][1]+2:]:
        reduced_app.append(ratio)

    return reduced_app

def reduct_to_original(reduced_app, BO_options):
    original_app = list()
    for idx,job_time in enumerate(reduced_app):
        if idx == BO_options['iteration_interval'][0]+1:
            original_app = original_app + \
                [job_time for i in range(BO_options['iteration_interval'][1]-BO_options['iteration_interval'][0]+1)]
        else:
            original_app.append(job_time)

    print(original_app)
    return original_app

def evaluate_knob(current_conf):
    
    rm_output_path(app)
    current_conf_dict = {}
    for knob_name, knob_value in zip(knob_domain_dict.keys(), current_conf):
        current_conf_dict[knob_name] = knob_value

    log_name = run(app_jar_path, app_log_path, spark_home, current_conf_dict)
    print("logname:" + log_name, end=" ")
    if log_name == "failure":
        # print("current_conf", current_conf, "y:", BO_options['upperbound_y'])

        y_entire_vanilla.append(BO_options['upperbound_y']) # for the save of the entire ys
        return BO_options['upperbound_y']
    else:
        log = log_analyzer(app_log_path + log_name)
        if log.failed:
            y = BO_options['upperbound_y']
        else:
            line = None

            if BO_options['granularity'] == 'job':
                y_list = [log.spark_init_time] + [job_time for job_time in log.job_execution_time.values()]
            elif BO_options['granularity'] == 'stage':
                y_list = [log.spark_init_time] + [stage_time for stage_time in log.stage_execution_time.values()]
            # print(y_list)
            app_path_split = app_jar_path.split("/")[-1]
            sample_run_path = "./sample-runs/" + BO_options['granularity'] + "/" + app_path_split.split("_")[0]
            if not os.path.exists(sample_run_path):
                os.system("mkdir " + sample_run_path)

            sample_run_path = sample_run_path + "/" + app_path_split.split(".")[0] + ".csv"
            
            if not os.path.exists(sample_run_path):
                with open(sample_run_path, "w") as f:
                    wrt = csv.writer(f, delimiter=',')
                    wrt.writerow(y_list)
                    y = sum(y_list)
            else:
                with open(sample_run_path, "r") as f:
                    rdr = csv.reader(f, delimiter=',')
                    line = None
                    for row in rdr:
                        line = row
                        break
                    # print(line)
                    if len(y_list) == len(line):
                        y = sum(y_list)
                        with open(sample_run_path, "a") as fw:
                            wrt = csv.writer(fw, delimiter=',')
                            wrt.writerow(y_list)
                    else:
                        os.system("mv " + app_log_path + log_name + " ./problematic-logs/")
                        y = BO_options['upperbound_y']
        y_entire_vanilla.append(y) # for the save of the entire ys
        # print(current_conf, y)
        print(y)
        return y
    


def ys_from_log(app_log_path, log_name, xi_distribute_ratio, BO_options):
    if log_name == "failure": #failure without log
        print("lognamefailure")
        if BO_options['failure_train']:
            ys = [BO_options['upperbound_y'] * ratio for ratio in xi_distribute_ratio]
            
        else:
            ys = []
        original_ys = ys
    else:
        log = log_analyzer(app_log_path + log_name)
        if not log.failed:
            print("not failed")

            if BO_options['granularity'] == 'job':
                ys = list(log.job_execution_time.values())
                original_ys = [log.spark_init_time] + ys
                new_jobs = []
                if BO_options['job_granularity'] == len(ys):
                    new_jobs.append(sum(ys))
                else:
                    for i in range(BO_options['job_granularity'], len(ys), BO_options['job_granularity']):
                        new_jobs.append(sum(ys[i-BO_options['job_granularity']:i]))
                        if (i + BO_options['job_granularity']) >= len(ys):
                            new_jobs.append(sum(ys[i:len(ys)]))
                        
                    
                    
                ys = [log.spark_init_time] + new_jobs
            elif BO_options['granularity'] == 'stage':
                ys = [log.spark_init_time] + list(log.stage_execution_time.values())
            # print(len(ys))
            # print(len(xi_distribute_ratio))


            if BO_options['reduct_iter']:
                ys = make_iter1_to_real(ys, BO_options)
                print("hi")
            # print(len(ys))

            if len(original_ys) < len(xi_distribute_ratio) and not BO_options['partial_failure_train']: #intermediate job failure
                print("intermediate failure")
                if BO_options['failure_train']:
                    ys = [BO_options['upperbound_y'] * ratio for ratio in xi_distribute_ratio]
                else:
                    ys = []

            

        else:
            print("logfailed")
            if BO_options['failure_train']:
                ys = [BO_options['upperbound_y'] * ratio for ratio in xi_distribute_ratio]
            else:
                ys = []

    return ys
    

def fine_grained_BO(org_app_jar_path, app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options):
    Y, runwise_y_lists, acq_list = [], [], []
    log = None
    total_iter = 0
    
    
    dimensions = knob_domain_dict.values()
    space = normalize_dimensions(dimensions)

    space2 = skopt.Space(space)
    space2 = skopt.Space(normalize_dimensions(space2.dimensions))
    n_dims = space2.transformed_n_dims
    
    rng = check_random_state(None)
    
    #nu: smoothness
    #low length_scale: covariance increases -> low correlations
    # 23-12-01 added -------------------------------------
    app_path_split = org_app_jar_path.split("/")[-1]

    print("/" + app_path_split.split("_")[0]+ "/" + app_path_split.split(".")[0] + ".csv")
    xi_distribute_ratio = load_ratio_ratiofirst("./sample-runs/" + BO_options['granularity'] + "/" + app_path_split.split("_")[0]+ "/" + app_path_split.split(".")[0] + ".csv")
    
    #iteration interval indicates the job indices excluding resource allocation time. therefore, all the indices should be increased by 1
    if BO_options['reduct_iter']:
        xi_distribute_ratio = make_reduced_app(xi_distribute_ratio, BO_options)
    
    # print("len_xi:", len(xi_distribute_ratio))
    # print(xi_distribute_ratio)
    
    job_num_gran = 1 + int((len(xi_distribute_ratio)-1)/BO_options['job_granularity'])
    if ((len(xi_distribute_ratio)-1)%BO_options['job_granularity'])!=0:
        job_num_gran += 1
    
    print("calulated job num:", job_num_gran)
    jobwise_X = [[] for i in range(job_num_gran)]
    # history_jobwise_X = [[] for ratio in xi_distribute_ratio]
    jobwise_y_lists = [[] for i in range(job_num_gran)]
    # history_jobwise_y = [[] for job in xi_distribute_ratio]


    cov_amplitude = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 1000.0))
    kernel = Matern(
        length_scale=np.ones(n_dims),
        length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)
    base_model = GaussianProcessRegressor(kernel=cov_amplitude * kernel,
                                            normalize_y=True,
                                            noise="gaussian",
                                            n_restarts_optimizer=2,
                                            random_state=rng.randint(0, np.iinfo(np.int32).max))
    
    n_random = list(range(BO_options['n_random_iter']))



    next_xs = initial_point_generator(space=space, n_samples=BO_options['n_random_iter'], method="random")

    for next_x in next_xs:
        # next_x = initial_point_generator(space=space, n_samples=1, method="lhs")[0]
        current_conf_dict = {}
        for knob_name, knob_value in zip(knob_domain_dict.keys(), next_x):
            current_conf_dict[knob_name] = knob_value

        rm_output_path(app_jar_path.split("/")[-1])
        log_name = run(app_jar_path, app_log_path, spark_home, current_conf_dict)
        print("logname:" + log_name)
        ys = ys_from_log(app_log_path, log_name, xi_distribute_ratio, BO_options)
        print(len(ys))
        if len(ys) == 0: #failure_train == False
            n_random.append(1)
            continue
        

        runwise_y_lists.append(ys)
        for job_x, job_ys, y in zip(jobwise_X[:len(ys)], jobwise_y_lists[:len(ys)], ys):
            job_x.append(next_x)
            job_ys.append(y)

        if BO_options['partial_failure_train'] and len(ys) < job_num_gran: # for the synchronization between list 'Y' idx and list 'runwise_y_lists'. However, this entry should not be selected as Y_opt.
            Y.append(BO_options['upperbound_y']) # prevent the selection of this evaluation as the minimum.
        else:
            if BO_options['reduct_iter']:
                Y.append(sum(reduct_to_original(ys,BO_options)))
            else:
                Y.append(sum(ys))
               
        # print('current_conf:', next_x, 'execution_time:', Y[-1])
    



    total_iter = BO_options['n_random_iter']
    

    #history check
    if BO_options["learn_history"]:
        historyname = org_app_jar_path.split("/")[-1].split(".")[0] + "_" + BO_options["conf_space"]

        for job_idx, job in enumerate(jobwise_X[:len(jobwise_X)]):
            if not os.path.exists("./evaluation-history/" + historyname + "_" + str(job_idx) + ".csv"):
                print("there is no history! starting from the scratch!")
            with open("./evaluation-history/" + historyname + "_" + str(job_idx) + ".csv", "r") as f:
                rdr = csv.reader(f)
                for line in rdr:
                    line = [int(v.split(".")[0]) for v in line]
                    jobwise_X[job_idx].append(line[:len(dimensions)])
                    jobwise_y_lists[job_idx].append(line[len(dimensions):][0])



    for iter in range(BO_options["max_iter"] - BO_options["n_random_iter"]):
        # print("iteration", iter, "started:")
        # get the next configuration with the aggregated acquisition function
        Y_opt_idx = np.argmin(Y)
        # print("the number of ys:", len(Y))        

        surro_models = []


        # print(Y_opt_idx)
        # print(jobwise_X)
        # print("----------------------------")
        # print(jobwise_y_lists)
        for job_x, job_ys in zip(jobwise_X, jobwise_y_lists):
            est = clone(base_model)
            est.fit(space.transform(job_x), job_ys)
            surro_models.append(est)
        
        
        
        if BO_options['reduct_iter']:
            next_x = suggest(acquisition_function=new_AAEI,
                             space=space,
                             surro_model=surro_models,
                             opt_ys=runwise_y_lists[Y_opt_idx],
                             BO_options=BO_options,
                             xi_distribute_ratio=xi_distribute_ratio,
                             iteration_interval=BO_options['iteration_interval'])
        else:
            next_x = suggest(acquisition_function=new_AEI,
                             space=space,
                             surro_model=surro_models,
                             opt_ys=runwise_y_lists[Y_opt_idx],
                             BO_options=BO_options,
                             xi_distribute_ratio=xi_distribute_ratio,
                             iteration_interval=BO_options['iteration_interval'])
        
        current_conf_dict = {}
        for knob_name, knob_value in zip(knob_domain_dict.keys(), next_x):
            current_conf_dict[knob_name] = knob_value
            
        rm_output_path(app_jar_path.split("/")[-1])
        # print("run the application with the next x.........")
        log_name = run(app_jar_path, app_log_path, spark_home, current_conf_dict)
        ys = ys_from_log(app_log_path, log_name, xi_distribute_ratio, BO_options)
        if len(ys) == 0: #failure_train == False
            print("failure occurs and no train")
            total_iter += 1
            continue

        print(ys, sum(ys))
        runwise_y_lists.append(ys)
        for job_x, job_ys, y in zip(jobwise_X[:len(ys)], jobwise_y_lists[:len(ys)], ys):
            job_x.append(next_x)
            job_ys.append(y)

        if BO_options['partial_failure_train'] and len(ys) < job_num_gran: # for the synchronization between list 'Y' idx and list 'runwise_y_lists'. However, this entry should not be selected as Y_opt.
            Y.append(BO_options['upperbound_y'])
        else:
            if BO_options['reduct_iter']:
                Y.append(sum(reduct_to_original(ys,BO_options)))
            else:
                Y.append(sum(ys))
        
        # print('current_conf:', next_x, 'execution_time:', Y[-1])

        total_iter += 1
        
        # finish the BO when the process reached at the pre-defined termination condition
    if BO_options['save_history']:
        filename = org_app_jar_path.split("/")[-1].split(".")[0] + "_" + BO_options["conf_space"]
        for job_idx, job in enumerate(jobwise_X):
            with open("./evaluation-history" + "/" + filename + "_" + str(job_idx) + ".csv", 'w') as f:
                wrt = csv.writer(f)
                for x, ys in zip(jobwise_X[job_idx], jobwise_y_lists[job_idx]):
                    wrt.writerow(x + [ys])
        



    return jobwise_X[0][np.argmin(np.array(Y))], np.min(Y), total_iter, Y, acq_list
        

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    # core: min, max cores of a node
    # memory: min, max memory of a node
    # instances: [2, entire cores in the cluster]
    # parallelism: [2, max instances]
    knob_domain_dict = {'spark.executor.cores':[1, 30]\
                        , 'spark.executor.memory':[1, 128]\
                        , 'spark.executor.instances': [2, 246]\
                        # , 'spark.memory.fraction': [1, 9]\
                        , 'spark.default.parallelism': [2, 246*3]\
                        , 'spark.driver.cores': [1, 30]\
                        , 'spark.driver.memory': [1, 250]\
                        # , 'spark.driver.maxResultSize': [0, 4]\
                        , 'spark.executor.memoryOverhead': [1, 8]\
                        # , 'spark.files.maxPartitionBytes': [1, 4]\
                        # , 'spark.memory.storageFraction': [1, 9] \
                        # , 'spark.reducer.maxSizeInFlight': [1, 6]\
                        # , 'spark.shuffle.file.buffer': [1, 8]\
                        # , 'spark.shuffle.compress': [0, 1] \
                        # , 'spark.shuffle.spill.compress': [0, 1] \
                        }
    # knob_domain_dict = {'spark.executor.cores':[1, 30]\
    #                     , 'spark.executor.memory':[1, 128]\
    #                     , 'spark.executor.instances': [2, 246]\
    #                     , 'spark.memory.fraction': [1, 9]\
    #                     , 'spark.default.parallelism': [2, 246*3]\
    #                     , 'spark.driver.cores': [1, 30]\
    #                     , 'spark.driver.memory': [1, 250]\
    #                     , 'spark.driver.maxResultSize': [0, 4]\
    #                     , 'spark.executor.memoryOverhead': [1, 8]\
    #                     , 'spark.files.maxPartitionBytes': [1, 4]\
    #                     , 'spark.memory.storageFraction': [1, 9] \
    #                     , 'spark.reducer.maxSizeInFlight': [1, 6]\
    #                     , 'spark.shuffle.file.buffer': [1, 8]\
    #                     , 'spark.shuffle.compress': [0, 1] \
    #                     , 'spark.shuffle.spill.compress': [0, 1] \
    #                     }
    # iteration intervals below not consider resource allocation job.
    #LinearRegression_tiny_iter20 - 3, 24
    #LinearRegression_tiny_iter10 - 3, 14
    #LinearRegression_tiny_iter1 - 3, 5



    app_list = [] # format: app
    # app_list.append("Linear_large_iter10.jar")
    # app_list.append("Linear_large_iter30.jar")
    # app_list.append("Linear_large_iter40.jar")
    # app_list.append("Linear_gigantic_.jar")
    # app_list.append("Linear_huge_.jar")
    # app_list.append("LR_large_.jar")
    # app_list.append("Pagerank_gigantic_.jar")
    # app_list.append("Wordcount_huge_.jar")
    # app_list.append("NWeight_large_.jar")
    # app_list.append("Kmeans_large_.jar")
    app_list.append("LR_large_.jar")
    red_app = "Linear_small_iter1.jar"

    base_app_path = "/usr/local/ML-optimization/laboratory/apps/"
    app_log_path = "../logs/"
    spark_home = "/usr/local/spark/"

    #Upperbound
    # LinearRegression_tiny_iter10 : 1e5
    # 
    # non_iteration = {"LinearRegression": {"iter20":(3, 24), "iter10":(3,-1)}, #iter20 - (3,24)
    #                  "LogisticRegression": {"iter20":()}(4, -2)} #iter20 - ()
    BO_options = {"total_trial":10,
                  "kernel": "matern", # matern+constant, RBF+constant
                  "max_iter": 30, # total iteration for BO
                  "n_random_iter": 3, # random initialization for BO
                  "sampler": "random", # sampling strategy for random initialization
                  "acquisition_func": "AEI",
                  "n_random_samples": 10000, # the number of random samples for acquisition function (minimize, L-BFGS)
                  "termination_EI": -1, # termination condition before max_iter
                  "target_app_time": -1,
                  "upperbound_y": 3000000.0,
                  "additional_minimize_failurehandling": False,
                  "additional_minimize": False,
                  "iteration_interval": (0, 45),
                  "reduced_iteration_interval": (3, 5),
                  "BO_list": {"vanilla":True, "fine":True, "fine-reduct":False},
                  "conf_space": "resource_only",
                  "granularity": "job",
                  "job_granularity": 1,
                  "partial_failure_train": True,
                  "failure_train": False,
                  "reduct_iter": False,
                  "learn_history": False,
                  "save_history": True
                  }
    
    print(BO_options)
    
    #app, red_app, total_trial, max_iter, n_random_iter, n_random_samples, upperbund_y
    for app in app_list:

        print(app)
        appdir = "./results/" + app.split(".")[0]

        if not os.path.exists(appdir):
            os.system("mkdir " + appdir)
        
        optiondir = appdir + "/" + str(BO_options["total_trial"]) + "-" + str(BO_options["max_iter"]) + "-" + str(BO_options["n_random_iter"])

        if not os.path.exists(optiondir):
            os.system("mkdir " + optiondir)
        
        space_optiondir = optiondir + "/" + BO_options["conf_space"]
        
        if not os.path.exists(space_optiondir):
            os.system("mkdir " + space_optiondir)
        
        granularity_optiondir = space_optiondir + "/" + BO_options["granularity"]
        
        if not os.path.exists(granularity_optiondir):
            os.system("mkdir " + granularity_optiondir)

        result_path = granularity_optiondir + "/"

        # print(result_path)

        final_x_fine = []
        final_y_fine = []
        final_opt_iter_fine = []
        final_opt_time_fine = []
        final_y_entire_fine = []
        
        final_x_vanilla = []
        final_y_vanilla = []
        final_opt_iter_vanilla = []
        final_opt_time_vanilla = []
        final_y_entire_vanilla = []

        final_x_reduct = []
        final_y_reduct = []
        final_opt_iter_reduct = []
        final_opt_time_reduct = []
        final_y_entire_reduct = []
        
        # print(BO_options['BO_list']['vanilla'])
        # if ("huge" or "gigantic") in app:
        #     BO_options['BO_list']['vanilla'] = True 
        # make_BO_init_data(app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options)

        vanilla_result_filename_opt = "opt_results_vanilla.txt"
        vanilla_result_filename_y = "y_results_vanilla.txt"
        
        fine_result_filename_opt = "opt_results_fine_second.txt"
        fine_result_filename_y = "y_results_fine_second.txt"
        
        if BO_options['BO_list']['vanilla']:
            if os.path.exists(result_path + vanilla_result_filename_opt):
                os.system("rm " + result_path + vanilla_result_filename_opt)
                
            if os.path.exists(result_path + vanilla_result_filename_y):
                os.system("rm " + result_path + vanilla_result_filename_y)
                
            app_jar_path = base_app_path  + app.split("_")[0] + "/target/scala-2.12/" + app
            print(app_jar_path)
            for idx in range(BO_options['total_trial']):
                print("-------------------------------------------------------------------------")
                print("gp_minimize trial", idx,"started")

                y_entire_vanilla = []
                rm_output_path(app)
                start = time.time()
                res = skopt.gp_minimize(evaluate_knob,
                                knob_domain_dict.values(),
                                acq_func="EI",
                                n_calls=BO_options['max_iter'],
                                n_initial_points=BO_options['n_random_iter'],
                                initial_point_generator="lhs",
                                acq_optimizer="sampling",
                                noise="gaussian")
                end = time.time()

                ############ get the actual execution time of the recommended X  ################################
                # current_conf_dict = dict()
                # for knob_name, knob_value in zip(knob_domain_dict.keys(), res.x):
                #     current_conf_dict[knob_name] = knob_value
                    
                # log_name = run(app_jar_path, app_log_path, spark_home, current_conf_dict)
                # log = log_analyzer(app_log_path + log_name)
            
                # current_Y_total = 0
                # current_Y_total += log.spark_init_time
                # for job_time in log.job_execution_time.values():
                #     current_Y_total += job_time
                #################################################################################################
                current_Y_total = np.min(res.func_vals)

                final_x_vanilla.append(res.x)
                final_y_vanilla.append(current_Y_total)
                final_opt_time_vanilla.append(end-start)
                
                with open(result_path + vanilla_result_filename_opt, "a") as f:
                    f.write(str(res.x) + "," + str(current_Y_total) + "," + str(end-start) + "\n")
                
                final_y_entire_vanilla.append(y_entire_vanilla)
                
                with open(result_path + vanilla_result_filename_y, "a") as f:
                    wtr = csv.writer(f)
                    wtr.writerow(y_entire_vanilla)
                
                print("optimized app time:", current_Y_total, end-start,"secs")
                print("-------------------------------------------------------------------------")

            # with open(result_path + "y_results_vanilla.txt", "w") as f:
            #     wtr = csv.writer(f)
            #     wtr.writerows(final_y_entire_vanilla)

            # with open(result_path + "opt_results_vanilla.txt", "w") as f:
            #     for x, y, t in zip(final_x_vanilla, final_y_vanilla, final_opt_time_vanilla):
            #         f.write(str(x) + "," + str(y) + "," + str(t) + "\n")
                    
            time.sleep(60)
        
        if BO_options['BO_list']['fine']:
            
            if os.path.exists(result_path + fine_result_filename_opt):
                os.system("rm " + result_path + fine_result_filename_opt)
                
            if os.path.exists(result_path + fine_result_filename_y):
                os.system("rm " + result_path + fine_result_filename_y)
                
            final_x_fine = []
            final_y_fine = []
            final_opt_iter_fine = []
            final_opt_time_fine = []
            final_y_entire_fine = []
            granularity = [1]
            for gran in granularity:
                BO_options['job_granularity'] = gran
                app_jar_path = base_app_path + app.split("_")[0] + "/target/scala-2.12/" + app

                for idx in range(BO_options['total_trial']):
                    print("-------------------------------------------------------------------------")
                    print("fine-grained trial", idx,"started")
                    
                    rm_output_path(app)
                    
                    start = time.time()
                    x, y, total_iter, y_entire, acq_list_fine = fine_grained_BO(app_jar_path, app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options)
                    # print(fine_grained_BO(app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options))
                    end = time.time()
                    final_x_fine.append(x)
                    final_y_fine.append(y)
                    final_opt_time_fine.append(end-start)
                    final_opt_iter_fine.append(total_iter)
                
                    with open(result_path + fine_result_filename_opt, "a") as f:
                        f.write(str(x) + "," + str(y) + "," + str(end-start) + "\n")
                        
                    final_y_entire_fine.append(y_entire)
                
                    with open(result_path + fine_result_filename_y, "a") as f:
                        wtr = csv.writer(f)
                        wtr.writerow(y_entire)
                        
                    print("fine_grained trial finished, optimized app time:", y, "optimization iter:", total_iter, "optimization time:", end-start,"secs")
                    print("-------------------------------------------------------------------------")
                
            
                # with open(result_path + "y_results_fine_secondtrial" + "_gran_" + str(gran) + ".txt", "w") as f:
                #     wtr = csv.writer(f)
                #     wtr.writerows(final_y_entire_fine)
                    
                # with open(result_path + "opt_results_fine_secondtrial" + "_gran_" + str(gran) + ".txt", "w") as f:
                #     for x, y, t in zip(final_x_fine, final_y_fine, final_opt_time_fine):
                #         f.write(str(x) + "," + str(y) + "," + str(t) + "\n")
                        
                
                time.sleep(60)
        
        if BO_options['BO_list']['fine-reduct']:
            # red_result_path = result_path + red_app.split(".")[0] + "/"
            # if not os.path.exists(red_result_path):
            #     os.system("mkdir " + red_result_path)
            red_app_jar_path = base_app_path + red_app.split("_")[0] + "/target/scala-2.12/" + red_app
            org_app_jar_path = base_app_path + app.split("_")[0] + "/target/scala-2.12/" + app

            for idx in range(BO_options['total_trial']):
                print("-------------------------------------------------------------------------")
                print("fine-grained-reduction trial", idx,"started")
                
                rm_output_path(app)
                start = time.time()
                x, y, total_iter, y_entire, acq_list_reduct = fine_grained_BO(org_app_jar_path, red_app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options)
                # print(fine_grained_BO(app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options))
                end = time.time()



                current_conf_dict = dict()

                ############ get the actual execution time of the recommended X  ################################
                for knob_name, knob_value in zip(knob_domain_dict.keys(), x):
                    current_conf_dict[knob_name] = knob_value
                    
                log_name = run(org_app_jar_path, app_log_path, spark_home, current_conf_dict)
                log = log_analyzer(app_log_path + log_name)
            
                
                ys = [log.spark_init_time] + list(log.job_execution_time.values())
                #################################################################################################

                final_x_reduct.append(x)
                final_y_reduct.append(sum(ys))
                final_opt_time_reduct.append(end-start)
                final_y_entire_reduct.append(y_entire)
                print("fine_grained-reduction trial finished, optimized app time:", sum(ys), "optimization iter:", total_iter, "optimization time:", end-start,"secs")
                print("-------------------------------------------------------------------------")
            
        
            with open(result_path + "y_results_fine_reduction.txt", "w") as f:
                wtr = csv.writer(f)
                wtr.writerows(final_y_entire_reduct)
                
            with open(result_path + "opt_results_reduct.txt", "w") as f:
                for x, y, t in zip(final_x_reduct, final_y_reduct, final_opt_time_reduct):
                    f.write(str(x) + "," + str(y) + "," + str(t) + "\n")

     
        print("fine-grained result:")
        for x, y, i, t in zip(final_x_fine, final_y_fine, final_opt_iter_fine, final_opt_time_fine):
            print(y, "ms of app execution time has found", i, "iterations of BO,", t, "secs of optimization time")
        
        print("vanilla result:")
        for x, y, i, t in zip(final_x_vanilla, final_y_vanilla, final_opt_iter_vanilla, final_opt_time_vanilla):
            print(y, "ms of app execution time has found", i, "iterations of BO,", t, "secs of optimization time")        
        
        
        # with open("results.txt", "w") as f:

        #     for x, y, t in zip(final_x_vanilla, final_y_vanilla, final_opt_time_vanilla):
        #         f.write(str(x) + "," + str(y) + "," + str(t) + "\n")
        #     for x, y, t in zip(final_x_fine, final_y_fine, final_opt_time_fine):
        #         f.write(str(x) + "," + str(y) + "," + str(t) + "\n")
        #     for x, y, t in zip(final_x_reduct, final_y_reduct, final_opt_time_reduct):
        #         f.write(str(x) + "," + str(y) + "," + str(t) + "\n")



        

