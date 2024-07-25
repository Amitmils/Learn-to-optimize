import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import Timer
import random

class PGA(nn.Module):
    def __init__(self,config,num_iter,pga_type='Classic'):
        super().__init__()
        self.config = config
        self.num_iter = num_iter
        if config.mu_matrix:
            tensor_shape = (num_iter, (config.B+1), max(config.M, config.L), max(config.L, config.N))
        else:
            tensor_shape = (num_iter, (config.B+1))

        mu = torch.zeros(tensor_shape)
        if pga_type == 'Classic' or True:
            mu +=50 * 1e-2
            # nn.init.normal_(mu, mean=50 * 1e-2, std=0.01)

        # nn.init.xavier_uniform_(mu)#normal_(mu, mean=0.0, std=0.01) # uniform_ xavier_uniform_(mu)
      
        self.hyp = nn.Parameter(mu)  # parameters = (mu_a, mu_(d,1), ..., mu_(d,B))
        self.pga_type = pga_type

    @Timer.timeit
    def forward(self, h,plot = False):
        # ------- Projection Gradient Ascent execution ---------
        # h - channel realization
        # num_of_iter - num of iters of the PGA algorithm
        wa,wd = self.init_variables(h)
        sum_rate = torch.zeros(self.num_iter, len(h[0]))
        for k in range(self.num_iter):
            wa, wd = self.perform_iter(h,wa,wd,sum_rate,k)
            Timer.new_iter = True
        sum_rate = torch.transpose(sum_rate, 0, 1)

        if plot:
            #only true for classical PGA
            plt.figure()
            y = [r.detach().numpy() for r in (sum(sum_rate.cpu())/h.shape[1])]
            x = np.array(list(range(self.num_iter))) +1
            plt.plot(x,y,marker='+',label=f'Classic ({y[-1]:.2f}',color='orange')
            plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the {self.pga_type} PGA')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Achievable Rate')
            plt.legend()
            plt.grid()
            plt.show()
        return sum_rate, wa, wd

    @Timer.timeit
    def init_variables(self,h):
        # svd for H_avg --> H = u*smat*vh
        _, _, vh = torch.linalg.svd(sum(h) / self.config.B, full_matrices=True)
        # initializing Wa as vh
        wa = vh[:, :, :self.config.L]
        wa = torch.cat(((wa[None, :, :, :],) * self.config.B), 0)
        # randomizing Wd,b
        wd = torch.randn(self.config.B, len(h[0]), self.config.L, self.config.N).to(wa.dtype) #makes it into complex128 if needed
        # projecting Wd,b onto the constraint
        wd = (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa @ wd, ord='fro')**2)))).reshape(len(h[0]), 1, 1) * wd

        self.prev_dWd = torch.zeros((self.config.B,h.shape[1],self.config.L,self.config.N),requires_grad=False,dtype=wa.dtype)

        return wa,wd

    def is_dWd_full_grad(self,iter,bin):
        if self.pga_type == 'Classic':
            return True

        if self.config.approx_dWd and iter in self.config.iters_to_approx:
            return False
        return True

    @Timer.timeit
    def perform_iter(self,h,wa,wd,sum_rate,iter_num):
        # ---------- Wa ---------------
        if self.config.mu_matrix:
            wa_t = wa + self.hyp[iter_num][0][:self.config.M,:self.config.L] * self.grad_wa(h, wa, wd) #gradient ascent
        else:
            wa_t = wa + self.hyp[iter_num][0] * self.grad_wa(h, wa, wd) #gradient ascent
        perform_proj = True or self.config.Wa_constrained or self.num_iter - 1 == iter_num
        wa = self.wa_projection(wa_t,wd,h,perform_proj)

        # ---------- Wd,b ---------------
        wd_t = wd.clone().detach()
        for i in range(self.config.B):
            full_grad = self.is_dWd_full_grad(iter_num,i)
            if self.config.mu_matrix:
                wd_t[i] = wd[i].clone().detach() + self.hyp[iter_num][i + 1][:self.config.L,:self.config.N] * self.grad_wd(h[i], wa[0], wd[i],full_grad,i) # gradient ascent
            else:
                wd_t[i] = wd[i].clone().detach() + self.hyp[iter_num][i + 1] * self.grad_wd(h[i], wa[0], wd[i],full_grad,i) # gradient ascent
        wd = self.wd_projection(wa,wd_t,h,True) #if perform_proj == False return wd_t


        # update the rate
        sum_rate[iter_num] = torch.real(self.calc_sum_rate(h, wa, wd)) #numerical error for imaginary - real to take it off
        return wa,wd

    @Timer.timeit
    def grad_wa(self, h, wa, wd):
        # calculates the gradient with respect to wa for a given channel (h) and precoders (wa, wd)
        if self.pga_type != 'Classic' and self.config.stoch_dWa:
            permutation = torch.randperm(self.config.B)[:self.config.Freq_bins_for_stoch_dWa]
            h = h[permutation]
            wa = wa[permutation]
            wd = wd[permutation]
        if self.pga_type == 'Classic' or (not(self.config.dWa_G_I) and not(self.config.dWa_G_Ones)):
            h_wa = h @ wa
            f2 = torch.mean(torch.transpose(h, 2, 3) @ torch.transpose(torch.linalg.inv(torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N))
                                                                             + h_wa @ wd @
                                                                             torch.transpose(wd, 2, 3).conj() @
                                                                             torch.transpose(h_wa, 2, 3).conj()), 2, 3)
                                                                             @ h_wa.conj() @ wd.conj() @
                                                                             torch.transpose(wd, 2, 3).conj(),axis=0)
        elif self.config.dWa_G_I:
            h_wa = h @ wa
            f2 = torch.mean(torch.transpose(h, 2, 3) @ torch.transpose(torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N)
                                                                             ), 2, 3)
                                                                             @ h_wa.conj() @ wd.conj() @
                                                                             torch.transpose(wd, 2, 3).conj(),axis=0)
        elif self.config.dWa_G_Ones:
            return torch.ones_like(wa)
        return torch.cat(((f2[None, :, :, :],) * self.config.B), 0)

    @Timer.timeit
    def wa_projection(self,wa_t,wd,h,perform_proj):
        if perform_proj:
            if self.config.Wa_constrained:
                mag = torch.abs(wa_t) + 1e-6
                return wa_t/mag
            else:
                return (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa_t @ wd, ord='fro') ** 2)))).reshape(len(h[0]), 1,                                                                                             1) * wa_t
        else:
            return wa_t
    @Timer.timeit
    def grad_wd(self, h, wa, wd,full_grad,freq_bin):
        # calculates the gradient with respect to wd,b for a given channel (h) and precoders (wa, wd)
        if full_grad:
            h_wa = h @ wa
            grad = (torch.transpose(h_wa, 1, 2) @
                torch.transpose(torch.linalg.inv(torch.eye(self.config.N).reshape((1, self.config.N, self.config.N)).repeat(len(h), 1, 1) +
                h_wa @ wd @ torch.transpose(wd, 1, 2).conj() @
                torch.transpose(h_wa, 1, 2).conj()), 1, 2) @
                h_wa.conj() @ wd.conj()) / self.config.B
            
            self.prev_dWd[freq_bin] = grad
            return grad
        else:
            return self.prev_dWd[freq_bin].clone().detach()
            return torch.ones_like(wd)
            h_wa = h @ wa
            return (torch.transpose(h_wa, 1, 2) @
               torch.eye(self.config.N).reshape((1, self.config.N, self.config.N)).repeat(len(h), 1, 1) @
                h_wa.conj() @ wd.conj()) / self.config.B

    @Timer.timeit
    def wd_projection(self,wa,wd_t,h,perform_proj):
        if perform_proj:
            return (torch.sqrt(self.config.N * self.config.B / (sum(torch.linalg.matrix_norm(wa @ wd_t, ord='fro') ** 2)))).reshape(len(h[0]),
                                                                                                              1,
                                                                                                              1) * wd_t
        else:
            return wd_t

    @Timer.timeit
    def calc_sum_rate(self, h, wa, wd):
        # calculates the rate for a given channel (h) and precoders (wa, wd)
        return torch.mean(torch.log((torch.eye(self.config.N).reshape((1, 1, self.config.N, self.config.N)) +
                       h @ wa @ wd @ torch.transpose(wd, 2, 3).conj() @
                       torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()).det()),axis=0)
    
