from knobs import load_knob_samples
from sparksubmit import sparksubmit_command, run
from log_analyzer import log_analyzer
import csv
import copy

def make_BO_init_data(app_jar_path, app_log_path, spark_home, knob_domain_dict, BO_options):
    knob_samples = load_knob_samples(BO_options)
    
    # print(knob_samples)
    
    for set_num, a_set in enumerate(knob_samples):
        set_results = []
        for conf in a_set:
            Y_actual = []
            int_conf = copy.deepcopy(conf)
            int_conf = [int(val) for val in int_conf]
            # print(int_conf)
            current_conf_dict = {}
            for knob_name, knob_value in zip(knob_domain_dict.keys(), int_conf):
                current_conf_dict[knob_name] = knob_value
            
            # print(int_conf)
            log_name = run(app_jar_path, app_log_path, spark_home, current_conf_dict, knob_domain_dict)
            log = log_analyzer(app_log_path + log_name, knob_domain_dict.keys())
            
            # print(int_conf)
            print(log.spark_init_time)
            print(log.job_execution_time.values())
            current_Y_total = 0
            current_Y_total += log.spark_init_time
            
            Y_actual.append(log.spark_init_time)
            
            for idx, job_time in enumerate(log.job_execution_time.values()):
                Y_actual.append(job_time)
                current_Y_total += job_time
                
            # scheme: total, init time, job0, ...., jobn
            
            int_conf.append(current_Y_total)
            set_results.append(int_conf + Y_actual)
            
        with open("set" + str(set_num) + ".csv", "w") as f:
            csvwriter = csv.writer(f)
            for result in set_results:
                csvwriter.writerow(result)
            f.close()