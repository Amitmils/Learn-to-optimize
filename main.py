
import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd
from utils import Timer,CONFIG,set_device
from PGA import PGA
from Unfolded_PGA import Unfolded_PGA
import os
import scipy.io as io
# matplotlib.use('Agg')



if __name__ == "__main__":
    config = CONFIG('config.yaml')
    config.device = set_device(config.use_cuda and config.train) #if its not training, use CPU


    if config.dataset_type == "QUAD":
        H_all_channels = torch.tensor(io.loadmat(config.Quad_data_path)['H']).to(config.device).transpose(0,1).transpose(3,0).transpose(2,3).to(torch.complex128)
        H_train = H_all_channels[:,:config.train_size]
        H_val = H_all_channels[:,config.train_size:config.train_size + config.valid_size]
        H_test = H_all_channels[:,config.train_size + config.valid_size:]

    elif config.dataset_type == "Rayleigh":
        if config.create_IID_dataset:
            # ---- Create Datasets ----
            print("\n\nGenerate Dataset...")
            H_train = torch.randn(config.B, config.train_size, config.N, config.M)
            H_val = torch.randn(config.B, config.valid_size, config.N, config.M)
            H_test = torch.randn(config.B, config.test_size, config.N, config.M)
            torch.save([H_train,H_val,H_test],config.Rayleigh_data_path)
        else:
            print("\n\nLoaded Dataset...")
            H_train,H_val,H_test = torch.load(config.Rayleigh_data_path)
            H_train = H_train.to(config.device)
            H_val = H_val.to(config.device)
            H_test = H_test.to(config.device)

    config.B = H_train.shape[0]
    config.N = H_train.shape[2]
    config.M = H_train.shape[3]

    #because i forgot to change config while changing datasets
    if config.B == 64:
        config.L = 12
        print(f"IDIOT PROOF L ACTIVATED! L = {config.L}")
    if config.B == 8:
        config.L = 10
        print(f"IDIOT PROOF L ACTIVATED! L = {config.L}")

    print(f"Train Set Size : {H_train.shape[1]}")
    print(f"Val Set Size : {H_val.shape[1]}")
    print(f"Test Set Size : {H_test.shape[1]}")
    print(f"B = {config.B}, N = {config.N}, M = {config.M} , L = {config.L}\n\n")

    # ---- Classical PGA ----
    classic_model = PGA(config,config.num_of_iter_pga,pga_type='Classic')
    Timer.enabled = True
    sum_rate_class, wa, wd = classic_model.forward(H_test,plot=False)
    sum_rate_class =sum_rate_class.detach().cpu()
    Timer.save_time_telemetry(save_path="time_telemetry.csv")

    Timer.enabled = False    
    num_trials = 1
    Total_Summary = ""
    for trial_num in range(num_trials):
        for k in [5]:
            # ---- Unfolded PGA ----
            unfolded_model = Unfolded_PGA(config)
            if config.train:
                train_losses,valid_losses = unfolded_model.train(H_train,H_val)
            sum_rate_unfold = unfolded_model.eval(H_test, plot = False)
            trial_summary = f"AVG Rate Per Iter : {sum(sum_rate_unfold)/sum_rate_unfold.shape[0]} STD Rate Per Iter : {torch.std(sum_rate_unfold,dim=0)} {unfolded_model.run_name}"
            Total_Summary = Total_Summary + "\n" + trial_summary
            print(trial_summary)
            plt.figure()
            plt.title(f"{config.dataset_type} Channel, W, Trial = {trial_num} \n Loss = {config.loss}")
            plt.plot(range(1,sum_rate_unfold.shape[1]+1),sum(sum_rate_unfold)/sum_rate_unfold.shape[0],marker='*',label=f'Unfolded ({(sum(sum_rate_unfold)/sum_rate_unfold.shape[0])[-1]:.2f},{torch.std(sum_rate_unfold,dim=0)[-1].item():.2f})')
            plt.plot(range(1,sum_rate_class.shape[1]+1),[r for r in (sum(sum_rate_class)/sum_rate_class.shape[0])],marker='+',label=f'Classic ({(sum(sum_rate_class)/sum_rate_class.shape[0])[-1]:.2f},{torch.std(sum_rate_class,dim=0)[-1].item():.2f})')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Achievable Rate')
            plt.legend()
            plt.savefig(os.path.join(unfolded_model.run_folder,"Test_Result.png"))
    print(f"\n\nTotal Summary :\n{Total_Summary}")


