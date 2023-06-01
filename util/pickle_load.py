import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


test_case = "mlp_mse"
# cwd_data = os.path.abspath(os.getcwd()) + "/experiment/data/result_data/"
# result_path = 'D:\Jupyter_Projs\W\Encoding_BO\experiment\data\\result_data\Hartmann6' + '/TargetMean_trials_20_budget_200_duration2410'
file_name = 'CoCaBO_1_best_vals_LCB_ARD_False_mix_4979'
result_path = r'D:\Jupyter_Projs\W\Encoding_BO\experiment\data\result_data\mlp_dia' + '/' + file_name

# tpe_name = r'TPE_trials_10_budget_200_duration27032'
# tpe_pth = r'D:\Jupyter_Projs\W\Encoding_BO\experiment\data\\result_data\xgb_accu' + '/' + tpe_name
# with open(tpe_pth, "rb") as f:
#     # agg_t, agg_b, agg_m, agg_s = get_m_s_encoding(pickle.load(f))
#     tpe_list = pickle.load(f)

with open(result_path, "rb") as f:
    # agg_t, agg_b, agg_m, agg_s = get_m_s_encoding(pickle.load(f))
    result_list = pickle.load(f)

result_list_new = result_list

# for ii in range(len(result_list[5][:200].values)):
#     if ii > 50:
#         result_list[5][]
#
# for i in range(20):
#     result_list_new[i][0] = result_list_new[i][0].item()


new_file_name = result_path + '_new'
output = open(new_file_name, 'wb')
pickle.dump(result_list_new, output)
output.close()
