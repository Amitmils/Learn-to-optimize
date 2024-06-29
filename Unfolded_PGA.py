import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import Timer
from PGA import ProjGA

class Unfolded_PGA():
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.PGA = ProjGA(config,config.num_of_iter_pga_unf,enable_timing=False)
        self.optimizer = torch.optim.Adam(self.PGA.parameters(), lr=self.config.lr)
        Timer.enabled = False

    def train(self,H_train,H_val):
        self.PGA.train()
        train_losses, val_losses = list(),list()
        for i in range(self.config.epochs):
            print(f"epoch {i}")
            H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[1]))]
            for b in range(0, len(H_train), self.config.batch_size):
                H = torch.transpose(H_shuffeld[b:b+self.config.batch_size], 0, 1)
                __, wa, wd = self.PGA.forward(H,self.config.num_of_iter_pga_unf)
                loss = self.sum_loss(wa, wd, H, self.config.batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # train loss
            __, wa, wd = self.PGA.forward(H_train ,self.config.num_of_iter_pga_unf)
            train_losses.append(self.sum_loss(wa, wd, H_train, self.config.train_size))

            # validation loss
            __, wa, wd = self.PGA.forward(H_val, self.config.num_of_iter_pga_unf)
            val_losses.append(self.sum_loss(wa, wd, H_val, self.config.valid_size))
        self.plot_learning_curve(train_losses,val_losses)
        return train_losses,val_losses
    
    def eval(self,H_test):
        self.PGA.eval()
        sum_rate_unf, __, __ = self.PGA.forward(H_test, self.config.num_of_iter_pga_unf)
        plt.figure()
        y = [r.detach().numpy() for r in (sum(sum_rate_unf)/self.config.test_size)]
        x = np.array(list(range(self.config.num_of_iter_pga_unf))) +1
        plt.plot(x, y, 'o')
        plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the unfolded PGA')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Achievable Rate')
        plt.grid()
        plt.show()

    def plot_learning_curve(self,train_losses,val_losses):
        y_t = [r.detach().numpy() for r in train_losses]
        x_t = np.array(list(range(len(train_losses))))
        y_v = [r.detach().numpy() for r in val_losses]
        x_v = np.array(list(range(len(val_losses))))
        plt.figure()
        plt.plot(x_t, y_t, 'o', label='Train')
        plt.plot(x_v, y_v, '*', label='Valid')
        plt.grid()
        plt.title(f'Loss Curve, Num Epochs = {config.epochs}, Batch Size = {config.batch_size} \n Num of Iterations of PGA = {config.num_of_iter_pga_unf}')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.show()

    def sum_loss(self,wa, wd, h, batch_size):
        a1 = torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()
        a2 = torch.transpose(wd, 2, 3).conj() @ a1
        a3 = h @ wa @ wd @ a2
        g = torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N)) + a3  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
        s = torch.log(g.det())  # s = log(det(g))
        ra = sum(s) / self.config.B
        loss = sum(ra) / batch_size
        return -loss
