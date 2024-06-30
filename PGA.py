import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import Timer

class PGA(nn.Module):
    def __init__(self,config,num_iter,pga_type='Classic'):
        super().__init__()
        self.config = config
        mu= torch.tensor([[50 * 1e-2] * (config.B+1)] *num_iter, requires_grad=True)
        self.hyp = nn.Parameter(mu)  # parameters = (mu_a, mu_(d,1), ..., mu_(d,B))
        self.pga_type = pga_type

    @Timer.timeit
    def forward(self, h, num_of_iter,plot = False):
        # ------- Projection Gradient Ascent execution ---------
        # h - channel realization
        # num_of_iter - num of iters of the PGA algorithm
        wa,wd = self.init_variables(h)
        sum_rate = torch.zeros(num_of_iter, len(h[0]))
        for k in range(num_of_iter):
            wa, wd = self.perform_iter(h,wa,wd,sum_rate,k)
            Timer.new_iter = True
        sum_rate = torch.transpose(sum_rate, 0, 1)

        if plot:
            #only true for classical PGA
            plt.figure()
            y = [r.detach().numpy() for r in (sum(sum_rate)/h.shape[1])]
            x = np.array(list(range(num_of_iter))) +1
            plt.plot(x, y, 'o')
            plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the {self.pga_type} PGA')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Achievable Rate')
            plt.grid()
            plt.show()
        return sum_rate, wa, wd

    @Timer.timeit
    def init_variables(self,h):
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
        return wa,wd

    @Timer.timeit
    def perform_iter(self,h,wa,wd,sum_rate,iter_num):
        # ---------- Wa ---------------
        wa_t = wa + self.hyp[iter_num][0] * self.grad_wa(h, wa, wd) #gradient ascent
        wa = self.wa_projection(wa_t,wd,h)

        # ---------- Wd,b ---------------
        wd_t = wd.clone().detach()
        for i in range(self.config.B):
            wd_t[i] = wd[i].clone().detach() + self.hyp[iter_num][i + 1] * self.grad_wd(h[i], wa[0], wd[i].clone().detach()) # gradient ascent
            wd = self.wd_projection(wa,wd_t,h)

        # update the rate
        sum_rate[iter_num] = self.calc_sum_rate(h, wa, wd)
        return wa,wd

    @Timer.timeit
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

    @Timer.timeit
    def wa_projection(self,wa_t,wd,h):
        return (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa_t @ wd, ord='fro') ** 2)))).reshape(len(h[0]), 1,
                                                                                                          1) * wa_t

    @Timer.timeit
    def grad_wd(self, h, wa, wd):
        # calculates the gradient with respect to wd,b for a given channel (h) and precoders (wa, wd)
        return (torch.transpose(wa, 1, 2) @ torch.transpose(h, 1, 2) @
                torch.transpose(torch.linalg.inv(torch.eye(self.config.N).reshape((1, self.config.N, self.config.N)).repeat(len(h), 1, 1) +
                h @ wa @ wd @ torch.transpose(wd, 1, 2).conj() @
                torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()), 1, 2) @
                h.conj() @ wa.conj() @ wd.conj()) / self.config.B

    @Timer.timeit
    def wd_projection(self,wa,wd_t,h):
        return (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa @ wd_t, ord='fro') ** 2)))).reshape(len(h[0]),
                                                                                                              1,
                                                                                                              1) * wd_t

    @Timer.timeit
    def calc_sum_rate(self, h, wa, wd):
        # calculates the rate for a given channel (h) and precoders (wa, wd)
        return sum(torch.log((torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N)) +
                       h @ wa @ wd @ torch.transpose(wd, 2, 3).conj() @
                       torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()).det())) / self.config.B
    
def test():
    tensor = torch.tensor([1.0, 2.0, 3.0])

    print(tensor.dtype)  # Should print torch.float32
    print(tensor.device)  # Should print cuda:0 if GPU is available