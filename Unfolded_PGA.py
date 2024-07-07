import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import Timer
from PGA import PGA
import torch.nn as nn
from datetime import datetime
import os

class Unfolded_PGA():
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        today = datetime.today()
        now = datetime.now()
        self.run_name = f"{today.strftime('D%d_M%m')}_{now.strftime('h%H_m%M')}___K_{config.num_of_iter_pga_unf}__loss_{config.loss}__dWaOnes_{config.Wa_G_Ones}__dWaI_{config.Wa_G_I}__dWd_{'-'.join([str(itr) for itr in config.full_grad_Wd_iter])}"
        self.run_folder = os.path.join("runs",self.run_name)
        os.makedirs(self.run_folder)
        self.PGA = PGA(config,config.num_of_iter_pga_unf,pga_type='Unfolded')
        self.optimizer = torch.optim.Adam(self.PGA.parameters(), lr=self.config.lr)
        Timer.enabled = False

    def train(self,H_train,H_val):
        self.PGA.train()
        train_losses, val_losses = list(),list()
        best_loss = torch.inf
        best_mu = 0
        for i in range(self.config.epochs):
            self.PGA.train()
            H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[1]))]
            for b in range(0, len(H_train), self.config.batch_size):
                H = torch.transpose(H_shuffeld[b:b+self.config.batch_size], 0, 1)
                sum_rate_in_batch_per_iter, wa, wd = self.PGA.forward(H)
                loss = self.calc_loss(sum_rate_in_batch_per_iter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.PGA.eval()
            with torch.no_grad():
                # train loss
                sum_rate_per_iter, wa, wd = self.PGA.forward(H_train)
                train_losses.append(self.calc_loss(sum_rate_per_iter))

                # validation loss
                sum_rate_per_iter, wa, wd = self.PGA.forward(H_val)
                val_losses.append(self.calc_loss(sum_rate_per_iter))

            if val_losses[-1] < best_loss:
                    best_loss = val_losses[-1]
                    best_loss_epoch = i
                    torch.save(self.PGA.state_dict(),os.path.join(self.run_folder,"PGA_model.pth"))
            
            print(f"{i} Loss Training : {train_losses[-1]:.2f} Loss Validation : {val_losses[-1]:.2f} ")
            print(f"Optimal MSE : {best_loss:.2f}  Epoch {best_loss_epoch}")

        self.plot_learning_curve(train_losses,val_losses)
        return train_losses,val_losses
    
    def eval(self,H_test,plot=True):
        self.PGA.load_state_dict(torch.load(os.path.join(self.run_folder,"PGA_model.pth")))
        self.PGA.eval()
        sum_rate_unf, __, __ = self.PGA.forward(H_test,plot=plot)
        sum_rate_unf = sum_rate_unf.detach().cpu()
        self.save_run_info(sum_rate_unf)
        return sum_rate_unf

    def plot_learning_curve(self,train_losses,val_losses):
        y_t = [r.detach().cpu().numpy() for r in train_losses]
        x_t = np.array(list(range(len(train_losses))))
        y_v = [r.detach().cpu().numpy() for r in val_losses]
        x_v = np.array(list(range(len(val_losses))))
        plt.figure()
        plt.plot(x_t, y_t, 'o', label='Train')
        plt.plot(x_v, y_v, '*', label='Valid')
        plt.grid()
        plt.title(f'Loss Curve, Num Epochs = {self.config.epochs}, Batch Size = {self.config.batch_size} \n Num of Iterations of PGA = {self.config.num_of_iter_pga_unf}, Loss = {self.config.loss}')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.run_folder,"Learning_Curve.png"))
    
    def save_run_info(self,sum_rate):
        with open(os.path.join(self.run_folder,"run_summary.txt"),'w') as file:
            for key,value in vars(self.config).items():
                file.write(f"{key} : {value}\n")
            file.write("\n\n")
            file.write(f"AVG Sum Rate Per Iter: {sum(sum_rate)/sum_rate.shape[0]}\n")
            file.write(f"STD Sum Rate Per Iter: {torch.std(sum_rate,dim=0)}\n")
            
    def calc_loss(self,sum_rate_per_iter,loss_iter = -1):
        if self.config.loss == 'one_iter':
            return -torch.mean(sum_rate_per_iter,dim=0)[loss_iter]

        elif self.config.loss == 'all_iter':
            _,num_iter = sum_rate_per_iter.shape
            weights =torch.log(torch.arange(2,2 + num_iter))
            loss = -torch.mean(sum_rate_per_iter * weights)

        elif self.config.loss == 'some_iter':
            weights =torch.log(torch.arange(2,2 + len(self.config.full_grad_Wd_iter)))
            loss = -torch.mean(sum_rate_per_iter[:,self.config.full_grad_Wd_iter] * weights)

        return loss

