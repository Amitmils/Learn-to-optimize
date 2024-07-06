import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import Timer
from PGA import PGA
import torch.nn as nn

class Unfolded_PGA():
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
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
                    torch.save(self.PGA.state_dict(), "best_PGA_model.pth")
            
            print(f"{i} Loss Training : {train_losses[-1]:.2f} Loss Validation : {val_losses[-1]:.2f} ")
            print(f"Optimal MSE : {best_loss:.2f}  Epoch {best_loss_epoch}")

        self.plot_learning_curve(train_losses,val_losses)
        return train_losses,val_losses
    
    def eval(self,H_test,plot=True):
        self.PGA.load_state_dict(torch.load("best_PGA_model.pth"))
        self.PGA.eval()
        sum_rate_unf, __, __ = self.PGA.forward(H_test,plot=plot)
        return sum_rate_unf.detach().cpu()

    def plot_learning_curve(self,train_losses,val_losses):
        y_t = [r.detach().cpu().numpy() for r in train_losses]
        x_t = np.array(list(range(len(train_losses))))
        y_v = [r.detach().cpu().numpy() for r in val_losses]
        x_v = np.array(list(range(len(val_losses))))
        plt.figure()
        plt.plot(x_t, y_t, 'o', label='Train')
        plt.plot(x_v, y_v, '*', label='Valid')
        plt.grid()
        plt.title(f'Loss Curve, Num Epochs = {self.config.epochs}, Batch Size = {self.config.batch_size} \n Num of Iterations of PGA = {self.config.num_of_iter_pga_unf}, Last Iter Loss = {self.config.loss_only_one_iter}')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

    def calc_loss(self,sum_rate_per_iter,loss_iter = -1):
        if self.config.loss_only_one_iter:
            return -torch.mean(sum_rate_per_iter,dim=0)[loss_iter]
        else:
            _,num_iter = sum_rate_per_iter.shape
            weights =torch.log(torch.arange(2,2+num_iter - 2))
            loss = -torch.mean(sum_rate_per_iter[:,[1,3,4]] * weights)
            return loss

