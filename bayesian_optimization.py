import numpy as np
import sys
sys.path.append("../utils")
from knobs import initial_point_generator, load_ratio_ratiofirst
from scipy.optimize import minimize, fmin_l_bfgs_b
from joblib import Parallel, delayed
from scipy.stats import norm
import math
import skopt

def suggest(acquisition_function, space, surro_model, opt_ys, BO_options, xi_distribute_ratio, iteration_interval):
    
    # _, candidate_x = get_knob_samples(space, knob_domain_dict.keys(), BO_options['n_random_samples'], 1, "random", "return", False)
    candidate_x = initial_point_generator(space=space, n_samples=100, method="lhs") \
        + initial_point_generator(space=space, n_samples=BO_options['n_random_samples']-100, method="random")
    
    
    # print(candidate_xs)
    BO_stop_flag = False
    failure_flag = False
    # get the list of EI for the candidates
    acq = acquisition_function(space.transform(candidate_x), opt_ys, surro_model, xi_distribute_ratio, iteration_interval)
        # print(test)
        
    def min_obj(X, opt_ys, surro_model, xi_distribute_ratio, iteration_interval): # sampled X (candidates)
        EI = acquisition_function(X, opt_ys, surro_model, xi_distribute_ratio, iteration_interval)
        return -EI
    
    results = []
    cand_x = None
    cand_acq = None
    
    if BO_options["additional_minimize"] == True: 
        new_candidate_x = [candidate_x[x] for x in np.argsort(acq)[-5:]]
        # for x, acq in zip(new_candidate_x, new_acquisition):
        #     print(space.transform([x]).reshape(1,-1))
        #     res = minimize(min_obj, x0=np.squeeze(space.transform([x])).reshape(1,-1), bounds=space.transformed_bounds, method='L-BFGS-B')
        #     print(res)
        #     results.append(res)
        # cand_x = np.array([space.inverse_transform(res.x.reshape(1,-1)) for res in results])
        # cand_acq = np.array([-res.fun for res in results])
        # print(np.array(new_candidate_x))
        results = Parallel(n_jobs=1)(
            delayed(fmin_l_bfgs_b)(
                min_obj, space.transform([x]),
                args=(opt_ys, surro_model, xi_distribute_ratio,iteration_interval),
                bounds=space.transformed_bounds,
                approx_grad=False,
                maxiter=20)
            for x in np.array(new_candidate_x))
        
        cand_x = np.array([r[0] for r in results])
        cand_acq = np.array([r[1] for r in results])

    else:
        cand_acq = acq
        cand_x = candidate_x
    
    next_x = cand_x[np.argmax(cand_acq)]
    best_acq = np.max(cand_acq)
    # print("next_x:", next_x)
    # print("best_acq:", best_acq)
    # print("cand_acq:", cand_acq)
    
    return next_x


def new_AEI(X, opt_ys, surro_models, xi_distribute_ratio, iteration_interval):
    aggregated_mu = 0
    aggregated_var = 0
    total_opt_time = 0
    for surro, opt_y, ratio in zip(surro_models, opt_ys, xi_distribute_ratio):
        if X.ndim == 1:
            mu, std = surro.predict([X], return_std=True)
        else:
            mu, std = surro.predict(X, return_std=True)
        aggregated_mu += mu
        aggregated_var += std*std
        total_opt_time += opt_y

    # print("------------------------------------------------------------------------------------")
    # print(aggregated_mu)
    # print(aggregated_var)

    values = np.zeros_like(aggregated_mu)
    aggregated_std = np.sqrt(aggregated_var)
    mask = aggregated_std > 0

    improve = total_opt_time - aggregated_mu[mask] - 0.01
    # print(improve)
    # print("------------------------------------------------------------------------------------")
    scaled = improve / aggregated_std[mask]

    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = aggregated_std[mask] * pdf
    values[mask] = exploit + explore

    return values

def new_AAEI(X, opt_ys, surro_models, xi_distribute_ratio, iteration_interval):
    aggregated_mu = 0
    aggregated_std = 0
    total_job_time = 0
    job_idx = 0
    # reducted_ratio = xi_distribute_ratio[:iteration_interval[0]+1] + xi_distribute_ratio[iteration_interval[1]+1:]
    for surro, job_time, ratio in zip(surro_models, opt_ys, xi_distribute_ratio):
        if X.ndim == 1:
            mu, std = surro.predict([X], return_std=True)
        else:
            mu, std = surro.predict(X, return_std=True)
        if job_idx == iteration_interval[0] + 1: #for resource allocation
            num_iter = (iteration_interval[1] - iteration_interval[0] + 1)
            aggregated_mu += mu * num_iter
            aggregated_std += std * std * num_iter
            total_job_time += job_time * num_iter
        else:
            aggregated_mu += mu
            aggregated_std += std * std
            total_job_time += job_time
        job_idx += 1
    
    values = np.zeros_like(aggregated_mu)
    aggregated_std = np.sqrt(aggregated_std)
    mask = aggregated_std > 0
    improve = total_job_time - aggregated_mu[mask]
    scaled = improve / aggregated_std[mask]

    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = aggregated_std[mask] * pdf
    values[mask] = exploit + explore
    
        
    return values
    


