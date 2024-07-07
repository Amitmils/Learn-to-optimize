
import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd
from utils import Timer,CONFIG,set_device
from PGA import PGA
from Unfolded_PGA import Unfolded_PGA
import os
matplotlib.use('Agg')



if __name__ == "__main__":
    config = CONFIG('config.yaml')
    config.device = set_device(config.use_cuda and config.train) #if its not training, use CPU

    if config.create_dataset:
        # ---- Create Datasets ----
        print("\n\nGenerate Dataset...")
        H_train = torch.randn(config.B, config.train_size, config.N, config.M)
        H_val = torch.randn(config.B, config.valid_size, config.N, config.M)
        H_test = torch.randn(config.B, config.test_size, config.N, config.M)
        torch.save([H_train,H_val,H_test],"Rayleigh_dataset.pt.pt")
    else:
          print("\n\nLoaded Dataset...")
          H_train,H_val,H_test = torch.load("Rayleigh_dataset.pt.pt")

    print(f"Train Set Size : {H_train.shape[1]}")
    print(f"Val Set Size : {H_val.shape[1]}")
    print(f"Test Set Size : {H_test.shape[1]}")
    print(f"B = {H_train.shape[0]}, N = {H_train.shape[2]}, M = {H_train.shape[3]}\n\n")

    # ---- Classical PGA ----
    classic_model = PGA(config,config.num_of_iter_pga,pga_type='Classic')
    Timer.enabled = True
    sum_rate_class, wa, wd = classic_model.forward(H_test,plot=False)
    sum_rate_class =sum_rate_class.detach().cpu()
    Timer.save_time_telemetry(save_path="time_telemetry.csv")
    
    num_trials = 2
    Total_Summary = ""
    trial_num = 0
    for trial_num in range(num_trials):
        for k in [5]:
            config.num_of_iter_pga_unf = k
            for trial_loss in [ "some_iter"]:
                    config.loss = trial_loss
                    # ---- Unfolded PGA ----
                    Timer.enabled = False    
                    unfolded_model = Unfolded_PGA(config)
                    if config.train:
                        train_losses,valid_losses = unfolded_model.train(H_train,H_val)
                    sum_rate_unfold = unfolded_model.eval(H_test, plot = False)
                    trial_summary = f"AVG Rate Per Iter : {sum(sum_rate_unfold)/sum_rate_unfold.shape[0]} STD Rate Per Iter : {torch.std(sum_rate_unfold,dim=0)} K = {config.num_of_iter_pga_unf} Trial = {trial_num} Full Grad Wd Iter {config.full_grad_Wd_iter} Loss = {config.loss}"
                    Total_Summary = Total_Summary + "\n" + trial_summary
                    print(trial_summary)
                    plt.figure()
                    plt.title(f"Rayleigh Channel, W, Trial = {trial_num} \n Loss = {config.loss}")
                    plt.plot(range(1,sum_rate_unfold.shape[1]+1),sum(sum_rate_unfold)/sum_rate_unfold.shape[0],marker='*',label=f'Unfolded ({(sum(sum_rate_unfold)/sum_rate_unfold.shape[0])[-1]:.2f},{torch.std(sum_rate_unfold,dim=0)[-1].item():.2f})')
                    plt.plot(range(1,sum_rate_class.shape[1]+1),[r for r in (sum(sum_rate_class)/sum_rate_class.shape[0])],marker='+',label=f'Classic ({(sum(sum_rate_class)/sum_rate_class.shape[0])[-1]:.2f},{torch.std(sum_rate_class,dim=0)[-1].item():.2f})')
                    plt.xlabel('Number of Iteration')
                    plt.ylabel('Achievable Rate')
                    plt.legend()
                    plt.savefig(os.path.join(unfolded_model.run_folder,"Test_Result.png"))
    print(f"\n\nTotal Summary :\n{Total_Summary}")


