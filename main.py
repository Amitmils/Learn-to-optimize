
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from utils import Timer,CONFIG,set_device
from PGA import PGA
from Unfolded_PGA import Unfolded_PGA


if __name__ == "__main__":
    config = CONFIG('config.yaml')
    config.device = set_device()

    # Create Datasets
    H_train = torch.randn(config.B, config.train_size, config.N, config.M)
    H_val = torch.randn(config.B, config.valid_size, config.N, config.M)
    H_test = torch.randn(config.B, config.test_size, config.N, config.M)

    # ---- Classical PGA ----
    classic_model = PGA(config,config.num_of_iter_pga)
    Timer.enabled = True
    sum_rate_class, wa, wd = classic_model.forward(H_test,config.num_of_iter_pga,plot=False)
    Timer.save_time_telemetery()
    # ---- Unfolded PGA ----
    Timer.enabled = False    
    unfolded_model = Unfolded_PGA(config)
    train_losses,valid_losses = unfolded_model.train(H_train,H_val)
    unfolded_model.eval(H_test)

