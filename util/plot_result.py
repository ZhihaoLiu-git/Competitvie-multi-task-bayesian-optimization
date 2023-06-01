import pickle
import numpy as np
import matplotlib.pyplot as plt
# set big font
import seaborn as sns
import os


var_cof = 0.08
def get_m_s_encoding(pickle_list):
    num_trial = len(pickle_list)
    budget = len(pickle_list[0])
    # budget = pickle_list[0]
    all_y = np.zeros(shape=[num_trial, budget])
    for id_trial, y in enumerate(pickle_list):
        all_y[id_trial, :] = y
    m = np.mean(all_y, axis=0)
    s = np.std(all_y, axis=0)
    # Avoid variance shadow widths that are too large 0.3*s
    return num_trial, budget, m, var_cof * s

def get_mu_std_coca(pickle_list):
    num_trial = len(pickle_list)
    budget = len(pickle_list[0])

    max_arr = np.array(pickle_list)
    m = np.mean(max_arr, axis=0)
    s = np.std(max_arr, axis=0)

    return num_trial, budget, m, var_cof * s


save_flag = True
# Ackley1C  Ackley1C17  Ackley5C  Ackley9C  twogaussian  svm_mse  svm_dia  mlp_mse mlp_dia Hartmann6 dim53Ackley michalewicz  xgb_accu pest
test_case = "xgb_accu"
cwd_data = os.path.abspath(os.getcwd()) + "/experiment/data/result_data/"
result_path = cwd_data + test_case + '/'
files = os.listdir(result_path)
result_dict = {}
# plt.tick_params(labelsize=20)
for f in files:
    # file_name = f"{result_path}{encoder}_trials_{max_trials}_budget_{budget}_duration{duration}"
    # split the file name by '_', intercept the substring before '_'
    if '_' in f:
        encoder_name = f[:f.index('_')]
        result_dict[encoder_name] = f

# sns.set_context("notebook", font_scale=1.8)
# sns.set_style("whitegrid")
# plt.style.use('fivethirtyeight')  # Plot Setting background

methods = ["Aggregate", "RandomOrder", "CoCaBO", "Onehot", "SMAC", "TPE", "TargetMean"]
colors = {"Aggregate": '#1f77b4', "RandomOrder": '#ff7f0e', "CoCaBO": '#2ca02c',
          "Onehot": '#d62728', "SMAC": '#9467bd', "TPE": '#8c564b', "TargetMean": '#17becf',
          '1': '#7f7f7f', "2": '#bcbd22', "3": '#e377c2'}

markers = {"Aggregate": "o", "RandomOrder": "s", "CoCaBO": "v",
           "Onehot": "^", "SMAC": "*", "TPE": "d", "TargetMean": "x",
           '1': "p", "2": "+", "3": "h"}

if "Aggregate" in result_dict.keys():
    with open(result_path + result_dict["Aggregate"], "rb") as f:
        agg_t, agg_b, agg_m, agg_s = get_m_s_encoding(pickle.load(f))
if "TargetMean" in result_dict.keys():
    with open(result_path + result_dict['TargetMean'], "rb") as f:
        mean_t, mean_b, mean_m, mean_s = get_m_s_encoding(pickle.load(f))
if 'Onehot' in result_dict.keys():
    with open(result_path + result_dict['Onehot'], "rb") as f:
        onehot_t, onehot_b, onehot_m, onehot_s = get_m_s_encoding(pickle.load(f))
if 'CoCaBO' in result_dict.keys():
    with open(result_path + result_dict['CoCaBO'], "rb") as f:
        coca_t, coca_b, coca_m, coca_s = get_mu_std_coca(pickle.load(f))
if 'SMAC' in result_dict.keys():
    with open(result_path + result_dict['SMAC'], "rb") as f:
        smac_t, smac_b, smac_m, smac_s = get_m_s_encoding(pickle.load(f))
if 'TPE' in result_dict.keys():
    with open(result_path + result_dict['TPE'], "rb") as f:
        tpe_t, tpe_b, tpe_m, tpe_s = get_m_s_encoding(pickle.load(f))
if 'RandomOrder' in result_dict.keys():
    with open(result_path + result_dict['RandomOrder'], "rb") as f:
        ran_t, ran_b, ran_m, ran_s = get_m_s_encoding(pickle.load(f))

# plot best function values
marker_size = 15
y_font_size = 60
x_font_size = 60
lw = 5
fig, ax = plt.subplots(figsize=(20, 12), tight_layout=True)
ax.set_xlabel("Function evaluations", fontsize=x_font_size)
# if test_case in ["mlp", "svm"]:
#     ax.set_ylabel("Negative MSE")
ax.set_ylabel("Best function value", fontsize=y_font_size)
if 'svm' in test_case or 'mlp' in test_case:
    ax.set_ylabel("Negative MSE(×100)", fontsize=y_font_size)
# if test_case in ["mlp", "svm"]:
#     ax.set_ylabel("Negative MSE", fontsize=40)
if test_case in ["xgb_accu"]:
    ax.set_ylabel("Accuracy", fontsize=y_font_size)

budget = 200
plt.margins(x=0.02, y=0.02)
plt.xticks(np.arange(0, budget+1, step=budget/4))
x_all_ticks = np.arange(budget+1)
if "Aggregate" in result_dict.keys():
    ax.plot(x_all_ticks, agg_m, linewidth=lw, label="Aggregate", color=colors['Aggregate'],
            marker=markers['Aggregate'], markersize=marker_size, markevery=10)
    ax.fill_between(x_all_ticks, agg_m - agg_s, agg_m + agg_s, color=colors['Aggregate'], alpha=0.2)
if "TargetMean" in result_dict.keys():
    ax.plot(np.arange(len(mean_m)), mean_m, linewidth=lw, label="TargetMean", color=colors['TargetMean'], marker=markers['TargetMean']
            , markersize=marker_size, markevery=10)
    ax.fill_between(np.arange(len(mean_m)), mean_m - mean_s, mean_m + mean_s, color=colors['TargetMean'], alpha=0.2)
if "Onehot" in result_dict.keys():
    ax.plot(np.arange(len(onehot_m)), onehot_m, linewidth=lw, label="Onehot", color=colors['Onehot'],
            marker=markers['Onehot'], markersize=marker_size, markevery=10)
    ax.fill_between(np.arange(len(onehot_m)), onehot_m - onehot_s, onehot_m + onehot_s, color=colors['Onehot'], alpha=0.2)
if "RandomOrder" in result_dict.keys():
    ax.plot(np.arange(len(ran_m)), ran_m, linewidth=lw, label="RandomOrder", color=colors['RandomOrder'],
            marker=markers['RandomOrder'], markersize=marker_size, markevery=10)
    ax.fill_between(np.arange(len(ran_m)), ran_m - ran_s, ran_m + ran_s, color=colors['RandomOrder'], alpha=0.2)

if "CoCaBO" in result_dict.keys():
    ax.plot(x_all_ticks, coca_m, linewidth=lw, label="CoCaBO", color=colors['CoCaBO'], marker=markers['CoCaBO']
            , markersize=marker_size, markevery=10)
    ax.fill_between(x_all_ticks, coca_m - coca_s, coca_m + coca_s, color=colors['CoCaBO'], alpha=0.2)
if "SMAC" in result_dict.keys():
    ax.plot(x_all_ticks, smac_m, linewidth=lw, label="SMAC", color=colors['SMAC'], marker=markers['SMAC']
            , markersize=marker_size, markevery=10)
    ax.fill_between(x_all_ticks, smac_m - smac_s, smac_m + smac_s, color=colors['SMAC'], alpha=0.2)
if "TPE" in result_dict.keys():
    ax.plot(x_all_ticks, tpe_m, linewidth=lw, label="TPE", color=colors['TPE'], marker=markers['TPE']
            , markersize=marker_size, markevery=10)
    ax.fill_between(x_all_ticks, tpe_m - tpe_s, tpe_m + tpe_s, color=colors['TPE'], alpha=0.2)


# ax.set_title(f"{test_case}", fontsize=50)
# plt.legend(loc='upper left', prop={'size': 40})

# To avoid overwriting previous images, use a different filename each time you save
i = 0
filename = result_path + test_case + f"_{i}.pdf"
# fig.tight_layout()
while os.path.exists(filename):
    i += 1
    filename = result_path + test_case + f"_{i}.pdf"
plt.tick_params(labelsize=35)  #The font size of the axis scale
# plt.grid()
if save_flag:
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.savefig(filename, dpi=300)
plt.show()


