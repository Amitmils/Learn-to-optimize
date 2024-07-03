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
                sum_rate_in_batch_per_iter, wa, wd = self.PGA.forward(H,self.config.num_of_iter_pga_unf)
                loss = self.calc_loss(sum_rate_in_batch_per_iter)
                # loss = self.sum_loss(wa, wd, H, self.config.batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.PGA.eval()
            with torch.no_grad():
                # train loss
                sum_rate_per_iter, wa, wd = self.PGA.forward(H_train ,self.config.num_of_iter_pga_unf)
                train_losses.append(self.calc_loss(sum_rate_per_iter))
                # train_losses.append(self.sum_loss(wa, wd, H_train, self.config.train_size))

                # validation loss
                sum_rate_per_iter, wa, wd = self.PGA.forward(H_val, self.config.num_of_iter_pga_unf)
                val_losses.append(self.calc_loss(sum_rate_per_iter))
                # val_losses.append(self.sum_loss(wa, wd, H_val, self.config.valid_size))
                
            if val_losses[-1] < best_loss:
                    best_loss = val_losses[-1]
                    best_mu = self.PGA.hyp.detach()
                    best_loss_epoch = i
            
            print(f"{i} Loss Training : {train_losses[-1]:.2f} Loss Validation : {val_losses[-1]:.2f} ")
            print(f"Optimal MSE : {best_loss:.2f}  Epoch {best_loss_epoch}")

        self.PGA.hyp = nn.Parameter(best_mu)
        self.plot_learning_curve(train_losses,val_losses)
        return train_losses,val_losses
    
    def eval(self,H_test,plot=True):
        self.PGA.eval()
        sum_rate_unf, __, __ = self.PGA.forward(H_test, self.config.num_of_iter_pga_unf,plot=plot)
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
        plt.title(f'Loss Curve, Num Epochs = {self.config.epochs}, Batch Size = {self.config.batch_size} \n Num of Iterations of PGA = {self.config.num_of_iter_pga_unf}')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.show()

    def calc_loss(self,sum_rate_per_iter):
        _,num_iter = sum_rate_per_iter.shape
        weights =torch.log(torch.arange(2,2+num_iter))
        loss = -torch.mean(sum_rate_per_iter * weights)
        return loss

    def sum_loss(self,wa, wd, h, batch_size):
        a1 = torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()
        a2 = torch.transpose(wd, 2, 3).conj() @ a1
        a3 = h @ wa @ wd @ a2
        g = torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N)) + a3  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
        s = torch.log(g.det())  # s = log(det(g))
        ra = sum(s) / self.config.B
        loss = sum(ra) / batch_size
        return -loss
