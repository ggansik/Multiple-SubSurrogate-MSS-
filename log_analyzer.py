import os
import json

class log_analyzer:
    def __init__(self, log_file_path):
        log = open(log_file_path, encoding='utf-8')
        # print(log_file_path)
        self.app_name = ""
        self.app_id = ""
        self.app_execution_time = 9999999999999
        self.app_start_time = -1
        self.app_end_time = -1
        self.knobs = []
        self.spark_properties = {}
        
        self.largest_stage_id = -1
        self.num_jobs = -1
        self.num_stages = -1
        
        self.job_stage_rdd_tree = {}
        
        self.stage_execution_time = {}
        self.job_execution_time = {}
        self.spark_init_time = 9999999999999
        self.agg_stage_execution_time = {}
        self.failed = False

        cur_job_start_time = -1
        for line in log:
            event = ''
            try:
                event = json.loads(line)
                # if event['Event'] == 'SparkListenerJobStart':
                #     print("hello")
            except:
                print("there is a problematic line: ")
                print(line)
                continue
                
            # when an application is submitted, JVM, spark, hadoop(yarn, hdfs, etc.), system, classpath info are logged once with 'SparkListenerEnvironmentUpdate'
            if event['Event'] == 'SparkListenerJobStart':
                cur_job_start_time = event['Submission Time']
                # print("cur_job", cur_job_start_time)
                if self.spark_init_time == 9999999999999:
                    self.spark_init_time = cur_job_start_time - self.app_start_time
                if event['Job ID'] not in self.job_stage_rdd_tree.keys():
                    self.job_stage_rdd_tree[event['Job ID']] = {}
                    
                    for stage in event['Stage Infos']:
                        if stage['Stage ID'] not in self.job_stage_rdd_tree[event['Job ID']].keys():
                            self.job_stage_rdd_tree[event['Job ID']][stage['Stage ID']] = []
                            
                            for rdd in stage['RDD Info']:
                                if rdd['RDD ID'] not in self.job_stage_rdd_tree[event['Job ID']][stage['Stage ID']]:
                                    self.job_stage_rdd_tree[event['Job ID']][stage['Stage ID']].append(rdd['RDD ID'])
                        
            elif event['Event'] == 'SparkListenerEnvironmentUpdate': 
                self.spark_properties = event['Spark Properties']
                
                # for knob_name in knob_name_list:
                #     self.knobs.append(self.spark_properties[knob_name])  
            # elif event['Event'] == 'SparkListenerTaskEnd': # after a task in an executor is ended, this event occurs
            #     continue
        
            elif event['Event'] == 'SparkListenerJobEnd':
                if event['Job Result']['Result'] == 'JobSucceeded':
                    self.job_execution_time[event['Job ID']] = event['Completion Time'] - cur_job_start_time
                
            elif event['Event'] == 'SparkListenerStageCompleted':
                if event['Stage Info']['Stage ID'] > self.largest_stage_id:
                    self.largest_stage_id = event['Stage Info']['Stage ID']
                # print(event['Stage Info'])
                # self.stage_execution_time[event['Stage Info']['Stage ID']] = int(event['Stage Info']['Completion Time']) - int(event['Stage Info']['Submission Time'])
                self.stage_execution_time[event['Stage Info']['Stage ID']] = -1
            elif event['Event'] == 'SparkListenerApplicationStart':
                self.app_start_time = event['Timestamp']
                self.app_name = event['App Name']
                self.app_id = event['App ID']

            elif event['Event'] == 'SparkListenerApplicationEnd':
                self.app_end_time = event['Timestamp']
            
        # what if initialization was succeeded? the other job was failed?
        if self.spark_init_time == 9999999999999:
            self.failed = True
        self.app_execution_time = int(self.app_end_time) - int(self.app_start_time)
        # self.job_execution_time[-1] = self.spark_init_time
        for job in self.job_stage_rdd_tree.keys():
            self.agg_stage_execution_time[job] = 0
            for stage in self.job_stage_rdd_tree[job].keys():
                
                if stage in self.stage_execution_time.keys(): #skipped stage may be exist
                    self.agg_stage_execution_time[job] += self.stage_execution_time[stage]
        self.num_jobs = len(self.job_stage_rdd_tree.keys())
        
        
if __name__ == '__main__':
    
    log_name = "application_1709601605654_3218"
    log = log_analyzer("../old-logs/" + log_name)
    print(log.app_id)
    print(log.app_name)
    print('it takes', log.app_execution_time, 'ms')
    print("spark init time:", log.spark_init_time)
    print(log.job_execution_time, "total (init + job): ", log.spark_init_time+sum(log.job_execution_time.values()))
    print(log.stage_execution_time, "stage total (init + stage):", log.spark_init_time+sum(log.stage_execution_time.values()))

    for job_id, stages in log.job_stage_rdd_tree.items():
        print("job", job_id, end=": ")
        for stage_id, rdds in log.job_stage_rdd_tree[job_id].items():
            print(stage_id, end=', ')
        print("")
    # print(log.job_stage_rdd_tree)
    # print(log.stage_execution_time , "total (init + stage): ", log.spark_init_time+sum([stage_time for stage_time in log.stage_execution_time.values()]))
    print(log.num_jobs, 'jobs are executed')
    # print(log.knobs)
    # print(log.spark_properties)
    print(len(log.stage_execution_time), 'stages are executed (', log.largest_stage_id+1-len(log.stage_execution_time), 'skipped )')
    
    
