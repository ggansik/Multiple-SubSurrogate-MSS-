import os
import subprocess
import time

def sparksubmit_command(app_jar_path, spark_home, current_conf, knob_considered):
    bool_vals = ['true', 'false']
    result_sizes = ['200m', '500m', '1g', '2g', '4g']

    return_str = spark_home + "bin/spark-submit --master yarn --deploy-mode client"
    
    # application properties ---------------------------------------------------------------------
    if 'spark.executor.memory' in knob_considered:
        return_str += (' --conf spark.executor.memory=' + str(current_conf['spark.executor.memory']) + 'g')

    if 'spark.driver.cores' in knob_considered:
        return_str += (' --conf spark.driver.cores=' + str(current_conf['spark.driver.cores']))

    if 'spark.driver.memory' in knob_considered:
        return_str += (' --conf spark.driver.memory=' + str(current_conf['spark.driver.memory']) + 'g')
            
    if 'spark.driver.maxResultSize' in knob_considered:
        return_str += (' --conf spark.driver.maxResultSize=' + str(result_sizes[current_conf['spark.driver.maxResultSize']]))

    if 'spark.executor.memoryOverhead' in knob_considered:
        return_str += (' --conf spark.executor.memoryOverhead=' + str(current_conf['spark.executor.memoryOverhead']*384) + 'm')

    # shuffle behavior --------------------------------------------------------------------------------

    if 'spark.reducer.maxSizeInFlight' in knob_considered:
        return_str += (' --conf spark.reducer.maxSizeInFlight=' + str(current_conf['spark.reducer.maxSizeInFlight'] * 24) + 'm')

    if 'spark.shuffle.compress' in knob_considered:
        return_str += (' --conf spark.shuffle.compress=' + str(bool_vals[current_conf['spark.shuffle.compress']]))

    if 'spark.shuffle.file.buffer' in knob_considered:
        return_str += (' --conf spark.shuffle.file.buffer=' + str(current_conf['spark.shuffle.file.buffer'] * 16) + 'k')

    if 'spark.shuffle.spill.compress' in knob_considered:
        return_str += (' --conf spark.shuffle.spill.compress=' + str(bool_vals[current_conf['spark.shuffle.spill.compress']]))


    # Compression & Serialization ---------------------------------------------------------------------



    # Memory Management ---------------------------------------------------------------------
    
    if 'spark.memory.fraction' in knob_considered:
        return_str += (' --conf spark.memory.fraction=' + str(float(current_conf['spark.memory.fraction'])/10))

    if 'spark.memory.storageFraction' in knob_considered:
        return_str += (' --conf spark.memory.storageFraction=' + str(float(current_conf['spark.memory.storageFraction']/10)))

    # Execution behavior ---------------------------------------------------------------------
    
    if 'spark.executor.instances' in knob_considered:
        return_str += (' --conf spark.executor.instances=' + str(current_conf['spark.executor.instances']))

    if 'spark.executor.cores' in knob_considered:
        return_str += (' --conf spark.executor.cores=' + str(current_conf['spark.executor.cores']))

    if 'spark.default.parallelism' in knob_considered:
        return_str += (' --conf spark.default.parallelism=' + str(current_conf['spark.default.parallelism']))

    if 'spark.files.maxPartitionBytes' in knob_considered:
        return_str += (' --conf spark.files.maxPartitionBytes=' + str(current_conf['spark.files.maxPartitionBytes'] * 64) + 'm')
        
    # Networking ---------------------------------------------------------------------

    if "Kmeans" in app_jar_path:
        return_str += " --jars"
        return_str += " /root/.cache/coursier/v1/https/repo1.maven.org/maven2/org/apache/mahout/mahout-hdfs/0.10.2/mahout-hdfs-0.10.2.jar,"
        return_str += "/root/.cache/coursier/v1/https/repo1.maven.org/maven2/org/apache/mahout/mahout-mr/0.10.2/mahout-mr-0.10.2.jar,"
        return_str += "/root/.cache/coursier/v1/https/repo1.maven.org/maven2/org/apache/mahout/mahout-integration/0.10.2/mahout-integration-0.10.2.jar,"
        return_str += "/root/.cache/coursier/v1/https/repo1.maven.org/maven2/org/apache/mahout/mahout-math/0.10.2/mahout-math-0.10.2.jar,"
        # return_str += "/root/.m2/repository/org/apache/hadoop/hadoop-common/3.2.1/hadoop-common-3.2.1.jar"
    elif "NWeight" in app_jar_path:
        return_str += " --jars"
        return_str += " /root/.m2/repository/it/unimi/dsi/fastutil/6.5.15/fastutil-6.5.15.jar"
    return_str += (" " + app_jar_path)
    if "NWeight" in app_jar_path:
        return_str += (" " + str(current_conf['spark.default.parallelism']))
    print(return_str)

    return return_str

def run(app_jar_path, app_log_path, spark_home, current_conf):

    for path in os.listdir(app_log_path):
        if path.split("_")[0] == "application":
            os.system("mv " + app_log_path + path + " " + "../old-logs/")
            if len(os.listdir("../old-logs/")) > 50:
                os.system("rm " + "../old-logs/" + sorted(os.listdir("../old-logs"))[0])
                
            # os.system("rm " + app_log_path + path)
            continue
    #conduct a sample run.
    
    p = subprocess.Popen(sparksubmit_command(app_jar_path, spark_home, current_conf, current_conf.keys())\
                , stderr=subprocess.STDOUT\
                , stdout=subprocess.PIPE\
                , shell=True\
                , close_fds=True\
                , start_new_session=True)
    
    msg, errs = p.communicate(timeout=7200)
    # print(msg)
    # print(errs)
    # print(msg)
    time.sleep(5)
    # print(msg)
    # print(errs)
    ret_code = p.poll()
    # print(ret_code)
    log = ""
        # print("app finished successfully")
    try:
        log_name = os.listdir(app_log_path)[0]
    except:
        log_name = "failure"
            
    # else:
    #     print("app failed")
        

    return log_name