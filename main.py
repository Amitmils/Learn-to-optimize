
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import yaml
import json

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

class ProjGA(nn.Module):

    def __init__(self,config,num_iter):
        super().__init__()
        self.config = config
        mu= torch.tensor([[50 * 1e-2] * (config.B+1)] *num_iter, requires_grad=True)
        self.hyp = nn.Parameter(mu)  # parameters = (mu_a, mu_(d,1), ..., mu_(d,B))

    def forward(self, h, num_of_iter,plot = False):
        # ------- Projection Gradient Ascent execution ---------
        # h - channel realization
        # num_of_iter - num of iters of the PGA algorithm

        # ---- initializing variables
        # svd for H_avg --> H = u*smat*vh
        _, _, vh = np.linalg.svd(sum(h) / self.config.B, full_matrices=True)
        vh = torch.from_numpy(vh)
        # initializing Wa as vh
        wa = vh[:, :, :self.config.L]
        wa = torch.cat(((wa[None, :, :, :],) * self.config.B), 0)
        # randomizing Wd,b
        wd = torch.randn(self.config.B, len(h[0]), self.config.L, self.config.N)
        # projecting Wd,b onto the constraint
        wd = (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa @ wd, ord='fro')**2)))).reshape(len(h[0]), 1, 1) * wd

        # defining an array which holds the values of the rate of each iteration
        sum_rate = torch.zeros(num_of_iter, len(h[0]))

        # update equations
        for x in range(num_of_iter):
            # ---------- Wa ---------------
            # gradient ascent
            wa_t = wa + self.hyp[x][0] * self.grad_wa(h, wa, wd)
            # projection
            wa = (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa_t @ wd, ord='fro') ** 2)))).reshape(len(h[0]), 1,
                                                                                                          1) * wa_t

            # ---------- Wd,b ---------------
            wd_t = wd.clone().detach()
            for i in range(self.config.B):
                # gradient ascent
                wd_t[i] = wd[i].clone().detach() + self.hyp[x][i + 1] * self.grad_wd(h[i], wa[0], wd[i].clone().detach())
                # projection
                wd = (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa @ wd_t, ord='fro') ** 2)))).reshape(len(h[0]),
                                                                                                              1,
                                                                                                              1) * wd_t
            # update the rate
            sum_rate[x] = self.sum_rate(h, wa, wd)

        sum_rate = torch.transpose(sum_rate, 0, 1)
        if plot:
            #only true for classical PGA
            plt.figure()
            y = [r.detach().numpy() for r in (sum(sum_rate)/h.shape[1])]
            x = np.array(list(range(num_of_iter))) +1
            plt.plot(x, y, 'o')
            plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the Classical PGA')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Achievable Rate')
            plt.grid()
            plt.show()
        return sum_rate, wa, wd

    def sum_rate(self, h, wa, wd):
        # calculates the rate for a given channel (h) and precoders (wa, wd)
        return sum(torch.log((torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N)) +
                       h @ wa @ wd @ torch.transpose(wd, 2, 3).conj() @
                       torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()).det())) / self.config.B

    def grad_wa(self, h, wa, wd):
        # calculates the gradient with respect to wa for a given channel (h) and precoders (wa, wd)
        f2 = sum(torch.transpose(h, 2, 3) @ torch.transpose(torch.linalg.inv(torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N))
                                                                             + h @ wa @ wd @
                                                                             torch.transpose(wd, 2, 3).conj() @
                                                                             torch.transpose(wa, 2, 3).conj() @
                                                                             torch.transpose(h, 2, 3).conj()), 2, 3)
                                                                             @ h.conj() @ wa.conj() @ wd.conj() @
                                                                             torch.transpose(wd, 2, 3)) / self.config.B
        return torch.cat(((f2[None, :, :, :],) * self.config.B), 0)

    def grad_wd(self, h, wa, wd):
        # calculates the gradient with respect to wd,b for a given channel (h) and precoders (wa, wd)
        return (torch.transpose(wa, 1, 2) @ torch.transpose(h, 1, 2) @
                torch.transpose(torch.linalg.inv(torch.eye(self.config.N).reshape((1, self.config.N, self.config.N)).repeat(len(h), 1, 1) +
                h @ wa @ wd @ torch.transpose(wd, 1, 2).conj() @
                torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()), 1, 2) @
                h.conj() @ wa.conj() @ wd.conj()) / self.config.B

class Unfolded_ProjGA():
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.PGA = ProjGA(config,config.num_of_iter_pga_unf)
        self.optimizer = torch.optim.Adam(self.PGA.parameters(), lr=self.config.lr)

    def train(self,H_train,H_val):
        self.PGA.train()


        train_losses, val_losses = [], []
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


if __name__ == "__main__":
    config = CONFIG('config.yaml')

    # Create Datasets
    H_train = torch.randn(config.B, config.train_size, config.N, config.M)
    H_val = torch.randn(config.B, config.valid_size, config.N, config.M)
    H_test = torch.randn(config.B, config.test_size, config.N, config.M)

    # ---- Classical PGA ----
    classical_model = ProjGA(config,config.num_of_iter_pga)
    sum_rate_class, __, __ = classical_model.forward(H_test,config.num_of_iter_pga,plot=False)

    # ---- Unfolded PGA ----
    unfolded_model = Unfolded_ProjGA(config)
    train_losses,valid_losses = unfolded_model.train(H_train,H_val)
    unfolded_model.eval(H_test)

