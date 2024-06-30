
import json
import yaml
import time
import pandas as pd
import torch

class CONFIG():
    def __init__(self,config_path) -> None:
        self.parse_config(config_path)
    def parse_config(self,config_path):
        ext = config_path.split('.')[-1]
        assert ext == 'json' or ext == 'yaml', "Format Not Supported!"
        with open(config_path, 'r') as f:   
            if ext == 'yaml':
                data = yaml.safe_load(f)
            elif ext == 'json':
                data = json.load(f)
        for key,value in data.items():
            setattr(self,key,value)
        return data

class Timer:
    time_telemetry = list()
    enabled = False
    new_iter = True
    @staticmethod
    def timeit(func):
        def timed(*args, **kwargs):
            if Timer.enabled == True:
                if Timer.new_iter:
                    Timer.time_telemetry.append(dict())
                    Timer.new_iter = False
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                total_time = end_time - start_time
                Timer.time_telemetry[-1][f"{func.__name__}__{sum(func.__name__ in key for key in Timer.time_telemetry[-1].keys())}"] = total_time * 1e3 #ms
            else:
                result = func(*args, **kwargs)
            return result
        return timed
    
    @staticmethod
    def save_time_telemetry(save_path):
        columns = ['Description']
        data_per_iter = list()
        num_iterations = len(Timer.time_telemetry)
        total_init_variable_time = Timer.time_telemetry[0]['init_variables__0']
        total_forward_time = Timer.time_telemetry[-1]['forward__0']
        init_variable_row = ('init_variables',)+(0,) * 2 * num_iterations + (total_init_variable_time,f"{(100 * total_init_variable_time / total_forward_time):.2f}%")
        forward_time_row = ('forward',) + (0,) * 2* num_iterations + (total_forward_time,f"{(100 * total_forward_time / total_forward_time):.2f}%")
        for iter_num in range(num_iterations):
            run_timing_dict = Timer.time_telemetry[iter_num]
            total_iter_time = run_timing_dict[f'perform_iter__0'] #its 0 because per iter you only do an iter once :)
            iter_data = list()
            cum_wd_proj = 0
            cum_grad_wd = 0
            for func,time in run_timing_dict.items():
                if 'init_variables' in func or 'forward__0' in func:
                    #add these in the last summarizing column
                    continue
                if sum(func.split("__")[0] in func_name for func_name in run_timing_dict.keys()) == 1:
                    func = func.split("__")[0] #if only happens once remove the number
                iter_data.append((func,round(time,4),f"{(100 * time / total_iter_time):.2f}%"))
                if "wd_projection" in func:
                    cum_wd_proj += time
                if "grad_wd" in func:
                    cum_grad_wd +=time
                if "perform_iter" in func:
                    iter_data.append(("Total grad wd",round(cum_grad_wd,4),f"{(100 * cum_grad_wd / total_iter_time):.2f}%"))
                    iter_data.append(("Total wd projection",round(cum_wd_proj,4),f"{(100 * cum_wd_proj / total_iter_time):.2f}%"))
            data_per_iter.append(iter_data)
            columns.append(f'iter {iter_num} time [ms]')
            columns.append(f'iter {iter_num} % of iter')

        #list of common func calls in each iter
        func_calls = [func_data[0] for func_data in iter_data if 'init_variables' not in func_data[0] and 'forward' not in func_data[0]]
        time_summary = [init_variable_row]
        for func_name in func_calls:
            func_tuple = (func_name,)
            running_time_sum = 0
            for data in data_per_iter:
                for func_calls_data in data:
                    if func_calls_data[0] == func_name:
                        func_tuple += func_calls_data[1:]
                        running_time_sum += func_calls_data[1]
                        break
            func_tuple += (round(running_time_sum,4),f"{(100 * running_time_sum / total_forward_time):.2f}%")
            time_summary.append(func_tuple)
        time_summary.append(forward_time_row)
        columns.append(f'Total time [ms]')
        columns.append(f' % of forward')

        df = pd.DataFrame(time_summary, columns=columns)
        df.to_csv(save_path, index=False)

def set_device():
    if torch.cuda.is_available(): #set defaults before importing scripts
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        torch.set_default_dtype(torch.float32)  # Set default data type
        torch.set_default_device('cuda')  # Set default device (optional)
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
