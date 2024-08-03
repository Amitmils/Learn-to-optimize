import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd
from utils import Timer,CONFIG,set_device
from PGA import PGA
from Unfolded_PGA import Unfolded_PGA
import os
import scipy.io as io
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import json


def create_plot_data(dataset_path):
    pass

    config = CONFIG('config.yaml')
    config.train = False
    config.dataset_type = "QUAD"

    config.device = set_device(config.use_cuda)

    H_all_channels = torch.tensor(io.loadmat(dataset_path)['H']).to(config.device).transpose(0,1).transpose(3,0).transpose(2,3).to(torch.complex128)

    if os.path.basename(dataset_path) == 'H_2400Channels_64B_32M_12N.mat':
        H_test = H_all_channels[:,-200:]
        config.L = 12
        # Large Scale init Wa with V log2
        model_list = {
            r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)" : {'marker':'x','label':r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)",'path': 'Important_runs/log2/mu_matrix/QUAD_SET__64B__12N__32M__12L/D01_M08_h14_m49__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_64__dWdApprox_1_3/PGA_model.pth'},
            r"$\mu$ Scalar" : {'marker':'o','label':r"$\mu$ Scalar",'path': 'Important_runs/log2/mu_scalar/QUAD_SET__64B__12N__32M__12L/D01_M08_h15_m36__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_64__dWdApprox_False/PGA_model.pth'},
            r"Classic PGA" : {'marker':'^','label':r"Classic PGA",'path': '','num_iter':48},
            "plot_data": {'saved_data_path': os.path.join('plots_data','large_scale_data_initWa_V_log2_train.json'),'enable_zoom' : True,
            'start_snr':-5,'end_snr':5,'zoomed_in_start_snr':2,'zoomed_in_end_snr':4,
            'zoomed_in_min_rate' : 4,'zoomed_in_max_rate' :5}
        }

    elif os.path.basename(dataset_path) == 'H_1200Channels_8B_12M_6N.mat':
        H_test = H_all_channels[:,-100:]
        config.L = 10

        #Small Scale init Wa with V log2
        model_list = {
            r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)" : {'marker':'x','label':r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)", 'path': 'Important_runs/log2/mu_matrix/QUAD_SET__8B__6N__12M__10L/D02_M08_h14_m32__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_1_3/PGA_model.pth'},
            r"$dW_a$ Approx | $\mu$ Matrix" :{'marker':'*','label':r"$dW_a$ Approx | $\mu$ Matrix",'path': 'Important_runs/log2/mu_matrix/QUAD_SET__8B__6N__12M__10L/D02_M08_h18_m57__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_False/PGA_model.pth'},
            r"$\mu$ Matrix":{'marker':'o','label': r"$\mu$ Matrix",'path': 'Important_runs/log2/mu_matrix/QUAD_SET__8B__6N__12M__10L/D02_M08_h15_m46__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_8__dWdApprox_False/PGA_model.pth'},
            r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Scalar":{'marker':',','label':r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Scalar",'path': 'Important_runs/log2/mu_scalar/QUAD_SET__8B__6N__12M__10L/D02_M08_h16_m47__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_1_3/PGA_model.pth'},
            r"$dW_a$ Approx | $\mu$ Scalar" :{'marker':'+','label':r"$dW_a$ Approx | $\mu$ Scalar",'path': 'Important_runs/log2/mu_scalar/QUAD_SET__8B__6N__12M__10L/D02_M08_h19_m41__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_False/PGA_model.pth'},
            r"$\mu$ Scalar": {'marker':'o','label':r"$\mu$ Scalar",'path': 'Important_runs/log2/mu_scalar/QUAD_SET__8B__6N__12M__10L/D02_M08_h17_m22__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_8__dWdApprox_False/PGA_model.pth'},
            # r"Classic PGA" : {'marker':'^','label':r"Classic PGA",'path': '','num_iter':100},
            "plot_data": {'saved_data_path': os.path.join('plots_data','small_scale_data_initWa_V_log2_train.json'),'enable_zoom' : True,
                        'start_snr':-5,'end_snr':5,'zoomed_in_start_snr':2,'zoomed_in_end_snr':4,
                        'zoomed_in_min_rate' : 4,'zoomed_in_max_rate' :5}
        }
    else:
        assert False, "Invalid Dataset"

    config.B = H_test.shape[0]
    config.N = H_test.shape[2]
    config.M = H_test.shape[3]

    print(f"Test Set Size : {H_test.shape[1]}")
    print(f"B = {config.B}, N = {config.N}, M = {config.M} , L = {config.L}\n\n")

    plot_data = {**model_list['plot_data']}
    if not(os.path.exists(plot_data['saved_data_path'])):
        
        snr_list = list(range(plot_data['start_snr'],plot_data['end_snr']+1))
        plot_data['snr'] = snr_list
        plot_data['models'] = {}
        for model_name,model_data in model_list.items():
            if model_name == "plot_data":
                continue
            plot_data['models'][model_name] = {}
            config.eval_model = model_data['path']
            unfolded_model = Unfolded_PGA(config)
            sum_rate_per_snr = list()
            std_rate_per_snr = list()
            for snr in snr_list:
                scalar = 10**(snr/10)
                if model_data['path'] == '':
                    classic_model = PGA(config,model_data['num_iter'],pga_type='Classic',lr = 7 * 1e-2)
                    sum_rate  = classic_model.forward(scalar * H_test,plot=False)[0].detach()
                    avg_sum_rate = sum(sum_rate)/sum_rate.shape[0]
                    std_sum_rate = np.std(sum_rate,dim=0)
                    sum_rate_per_snr.append(max(avg_sum_rate).item()/config.N)
                    std_rate_per_snr.append(std_sum_rate[torch.argmax(avg_sum_rate)].item()/config.N)
                else:
                    avg_sum_rate,std_sum_rate = unfolded_model.eval(scalar * H_test, plot = False,verbose = False)
                    sum_rate_per_snr.append(avg_sum_rate[-1].item()/config.N)
                    std_rate_per_snr.append(std_sum_rate[-1].item()/config.N)

                if snr == 0:
                    print(f"{model_name} : {sum_rate_per_snr[-1]}")

            plot_data['models'][model_name]['avg_sum_rate_per_snr'] = {snr : sum_rate_per_snr[i] for i,snr in enumerate(snr_list)} #for json readability
            plot_data['models'][model_name]['std_sum_rate_per_snr'] = {snr : std_rate_per_snr[i] for i,snr in enumerate(snr_list)} #for json readability
            plot_data['models'][model_name]['label'] = model_data['label']
            plot_data['models'][model_name]['marker'] = model_data['marker']

        with open(plot_data['saved_data_path'], 'w') as file:
            json.dump(plot_data, file, indent=4)
            # yaml.dump(data, file, default_flow_style=False)
    else:
        with open(plot_data['saved_data_path'], 'r') as file:
            plot_data = json.load(file)
    return plot_data
    
if __name__ == '__main__':
    dataset_path = 'H_1200Channels_8B_12M_6N.mat' # H_2400Channels_64B_32M_12N H_1200Channels_8B_12M_6N

    plot_data = create_plot_data(dataset_path)
    #plot_data['zoomed_in_start_snr'] = 
    #plot_data['zoomed_in_end_snr'] = 
    #plot_data['zoomed_in_min_rate'] = 
    #plot_data['zoomed_in_max_rate'] = 

    fig, ax = plt.subplots()
    inset_ax = inset_axes(ax, width="20%", height="20%", loc='lower right',bbox_to_anchor=(0, 0.2, 1, 1), bbox_transform=ax.transAxes)
    # for model in plot_data['models'].values():
    #     ax.plot(plot_data['snr'], model['avg_sum_rate_per_snr'].values(), label=model['label'], marker=model['marker'])
    #     if plot_data['enable_zoom']:
    #         inset_ax.plot(plot_data['snr'],model['avg_sum_rate_per_snr'].values(), label=model['label'], marker=model['marker'])
    
    for model in plot_data['models'].values():
        ax.plot(list(map(int, model['avg_sum_rate_per_snr'].keys())), model['avg_sum_rate_per_snr'].values(), label=model['label'], marker=model['marker'])
        if plot_data['enable_zoom']:
            inset_ax.plot(list(map(int, model['avg_sum_rate_per_snr'].keys())),model['avg_sum_rate_per_snr'].values(), label=model['label'], marker=model['marker'])
    
    ax.set_xticks(plot_data['snr'])
    # ax.set_xticks(range(plot_data['start_snr'],plot_data['end_snr']+1))
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Achievable Rate')
    ax.grid()
    ax.legend()

    if plot_data['enable_zoom']:
        inset_ax.set_xlim(plot_data['zoomed_in_start_snr'],plot_data['zoomed_in_end_snr'])
        inset_ax.set_xticks(range(plot_data['zoomed_in_start_snr'],plot_data['zoomed_in_end_snr']+1))
        inset_ax.set_ylim(plot_data['zoomed_in_min_rate'], plot_data['zoomed_in_max_rate'])
        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5")
        inset_ax.grid()

    # with open(plot_data['saved_data_path'], 'wb') as file:
    #         json.dump(plot_data, file)
    plt.savefig(plot_data['saved_data_path'].replace('json','png'))
    plt.show()





    #Small Scale init Wa with V^H log10
    # model_list = [
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)",'marker':'x','path': 'Important_runs/init_wa_VH/mu_matrix/QUAD_SET__8B__6N__12M__10L/D24_M07_h21_m21__QUAD_SET__K_5__loss_one_iter__WaConst_True__dWaOnes__True__Q_8__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$dW_a$ Approx | $\mu$ Matrix",'marker':'*','path': 'Important_runs/init_wa_VH/mu_matrix/QUAD_SET__8B__6N__12M__10L/D24_M07_h18_m43__QUAD_SET__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"$\mu$ Matrix",'marker':'o','path': 'Important_runs/init_wa_VH/mu_matrix/QUAD_SET__8B__6N__12M__10L/D31_M07_h11_m19__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Scalar",'marker':',','path': 'Important_runs/init_wa_VH/mu_scalar/QUAD_SET__8B__6N__12M__10L/D31_M07_h12_m22__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$dW_a$ Approx | $\mu$ Scalar",'marker':'+','path': 'Important_runs/init_wa_VH/mu_scalar/QUAD_SET__8B__6N__12M__10L/D31_M07_h12_m57__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"$\mu$ Scalar",'marker':'o','path': 'Important_runs/init_wa_VH/mu_scalar/QUAD_SET__8B__6N__12M__10L/D31_M07_h13_m41__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"Classic PGA",'marker':'^','path': '','num_iter':15},
    # ]

    #Small Scale init Wa with V log10
    # model_list = [
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)",'marker':'x','path': 'Important_runs/init_wa_V/mu_matrix/QUAD_SET__8B__6N__12M__10L/D01_M08_h05_m46__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$dW_a$ Approx | $\mu$ Matrix",'marker':'*','path': 'Important_runs/init_wa_V/mu_matrix/QUAD_SET__8B__6N__12M__10L/D01_M08_h06_m44__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"$\mu$ Matrix",'marker':'o','path': 'Important_runs/init_wa_V/mu_matrix/QUAD_SET__8B__6N__12M__10L/D01_M08_h06_m53__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Scalar",'marker':',','path': 'Important_runs/init_wa_V/mu_scalar/QUAD_SET__8B__6N__12M__10L/D01_M08_h07_m05__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$dW_a$ Approx | $\mu$ Scalar",'marker':'+','path': 'Important_runs/init_wa_V/mu_scalar/QUAD_SET__8B__6N__12M__10L/D01_M08_h07_m12__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"$\mu$ Scalar",'marker':'o','path': 'Important_runs/init_wa_V/mu_scalar/QUAD_SET__8B__6N__12M__10L/D01_M08_h07_m21__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_8__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"Classic PGA",'marker':'^','path': '','num_iter':22},
    # ]




    # # Large Scale init Wa with V^H log10
    # model_list = [
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)",'marker':'x','path': 'Important_runs/init_wa_VH/mu_matrix/QUAD_SET__64B__12N__32M__12L/D25_M07_h11_m38__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_64__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$\mu$ Matrix",'marker':'o','path': 'Important_runs/init_wa_VH/mu_matrix/QUAD_SET__64B__12N__32M__12L/7500_epochs_D25_M07_h10_m12__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_64__dWdApprox_False/PGA_model.pth'},
    #     # {'name': r"Classic PGA",'marker':'^','path': ''},
    # ]
    # # Large Scale init Wa with V log10
    # model_list = [
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)",'marker':'x','path': 'Important_runs/init_wa_V/mu_matrix/QUAD_SET__64B__12N__32M__12L/D01_M08_h10_m15__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_64__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$\mu$ Scalar",'marker':'o','path': 'Important_runs/init_wa_V/mu_scalar/QUAD_SET__64B__12N__32M__12L/D01_M08_h11_m47__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_64__dWdApprox_False/PGA_model.pth'},
    #     # {'name': r"Classic PGA",'marker':'^','path': ''},
    # ]

    # Large Scale init Wa with V log2
    # model_list = [
    #     {'name': r"$dW_a$ Approx | $dW_{d.b}$ Approx | $\mu$ Matrix (APGA)",'marker':'x','path': 'Important_runs/log2/mu_matrix/QUAD_SET__64B__12N__32M__12L/D01_M08_h14_m49__K_5__loss_one_iter__WaConst_True__dWaOnes_True__Q_64__dWdApprox_1_3/PGA_model.pth'},
    #     {'name': r"$\mu$ Scalar",'marker':'o','path': 'Important_runs/log2/mu_scalar/QUAD_SET__64B__12N__32M__12L/D01_M08_h15_m36__K_5__loss_one_iter__WaConst_True__dWaOnes_False__Q_64__dWdApprox_False/PGA_model.pth'},
    #     {'name': r"Classic PGA",'marker':'^','path': '','num_iter':45},
    # ]






