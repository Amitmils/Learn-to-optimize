
import matplotlib.pyplot as plt
import torch
import pandas as pd
from utils import Timer,CONFIG,set_device
from PGA import PGA
from Unfolded_PGA import Unfolded_PGA


if __name__ == "__main__":
    config = CONFIG('config.yaml')
    config.device = set_device(config.use_cuda and config.train) #if its not training, use CPU

    # ---- Create Datasets ----
    H_train = torch.randn(config.B, config.train_size, config.N, config.M)
    H_val = torch.randn(config.B, config.valid_size, config.N, config.M)
    H_test = torch.randn(config.B, config.test_size, config.N, config.M)

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
        # for trial_loss in ["one_iter","all_iter","some_iter"]:
        #         config.loss = trial_loss
                # ---- Unfolded PGA ----
                Timer.enabled = False    
                unfolded_model = Unfolded_PGA(config)
                if config.train:
                    train_losses,valid_losses = unfolded_model.train(H_train,H_val)
                sum_rate_unfold = unfolded_model.eval(H_test, plot = False)
                trial_summary = f"Achievable Rate Per Iter : {sum(sum_rate_unfold)/sum_rate_unfold.shape[0]}  K = {config.num_of_iter_pga_unf} Trial = {trial_num} "#)Full Grad Wd Iter {config.full_grad_Wd_iter} Loss = {config.loss}"
                Total_Summary = Total_Summary + "\n" + trial_summary
                print(trial_summary)
                plt.figure()
                plt.title(f"Rayleigh Channel, K = {config.num_of_iter_pga_unf}, Trial = {trial_num}")# \n Loss = {config.loss}")
                plt.plot(range(1,sum_rate_unfold.shape[1]+1),sum(sum_rate_unfold)/sum_rate_unfold.shape[0],marker='*',label=f'Unfolded ({(sum(sum_rate_unfold)/sum_rate_unfold.shape[0])[-1]:.2f})')
                plt.plot(range(1,sum_rate_class.shape[1]+1),[r for r in (sum(sum_rate_class)/sum_rate_class.shape[0])],marker='+',label=f'Classic ({(sum(sum_rate_class)/sum_rate_class.shape[0])[-1]:.2f})')
                plt.xlabel('Number of Iteration')
                plt.ylabel('Achievable Rate')
                plt.legend()
    print(f"\n\nTotal Summary :\n{Total_Summary}")
    plt.show()


