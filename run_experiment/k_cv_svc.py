import os
import numpy as np
import pandas as pd
import GPy
import pickle
import random
from sklearn.svm import SVC, SVR
from sklearn import metrics, datasets
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from scipy.stats import qmc
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier

class GetCmtData:
    def __init__(self, tasks_list=[0.1, 0.5, 3, 5], dataset=None):
        self.tasks_list = tasks_list
        self.dataset = dataset

    def get_init_data(self, num_sampling, bounds=[[0, 1], [0, 1]], save_path=None, seed=0):
        assert save_path is not None
        # Verify that the saving folder exists, if not, then create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        all_data = []
        for i in range(len(self.tasks_list)):
            file_name = save_path + '//' + f"task{i}_init" + str(seed)
            # Verify that the data exists
            if os.path.exists(file_name):
                print(f"Using existing data for seed {seed}, ")
                with open(file_name, 'rb') as save_file:
                    all_data.append(pickle.load(save_file))

            else:
                print(f"Creating init data for seed {seed}, {file_name}")
                l_bounds = [b[0] for b in bounds]
                u_bounds = [b[1] for b in bounds]
                yinit = np.zeros([num_sampling, 1])
                # latinhypercube
                sampler = qmc.LatinHypercube(d=len(bounds), optimization='random-cd') #seed=
                sample = sampler.random(n=num_sampling)

                # solbol
                # sampler = qmc.Sobol(d=len(bounds), scramble=False, seed=seed)
                # sample = sampler.random_base2(m=2) # here, num_sampling is 2^m
                Xinit = qmc.scale(sample, l_bounds, u_bounds)
                for j in range(num_sampling):

                    yinit[j] = self.get_task_output(task_index=i, x=Xinit[j])

                task_data = {i: i, 'X': Xinit, 'Y': yinit}
                all_data.append(task_data)

                with open(file_name, 'wb') as data_file:
                    pickle.dump(task_data, data_file)

        return all_data

    def get_task_output(self, task_index=None, x=None):
        assert task_index is not None and x is not None
        X, y = self.dataset
        # clf = SVC(kernel="rbf", C=x[0], gamma=x[1])
        # scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
        # return scores.mean()
        # reg = SVR(kernel='rbf', C=x[0], gamma=x[1])
        # all_mse = cross_val_score(reg, X, y, cv=kf, scoring='neg_mean_squared_error')
        # return all_mse.mean()

        kf = KFold(n_splits=self.tasks_list[task_index], shuffle=True, random_state=42)
        clf = SVC(kernel='poly', C=x[0], gamma=x[1])
        scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
        return scores.mean()

def get_design_domain(bounds, num_sampling, save_path='', seed=0):
    file_name = save_path + f"designX{str(seed)}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Verify that the data exists
    if os.path.exists(file_name):
        print(f"Using existing data for seed {seed}, ")
        with open(file_name, 'rb') as save_file:
            designX = pickle.load(save_file)
    else:
        print(f"Creating init data for seed {seed}, {file_name}")
        # using latine hypercube sampling
        l_bounds = [b[0] for b in bounds]
        u_bounds = [b[1] for b in bounds]
        sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
        sample = sampler.random(n=num_sampling)
        designX = qmc.scale(sample, l_bounds, u_bounds)

    with open(file_name, 'wb') as data_file:
        pickle.dump(designX, data_file)

    return designX

def select_dataset(name):
    if name == 'generate_classification':
        # n_samples: How many observations do you want to generate?
        # n_features: The number of numerical features.
        # n_informative: The number of features that are ‘useful.’ Only these features carry the signal that your model will use to classify the dataset.
        # n_classes: The number of unique classes (values) for the target label.
        # n_clusters_per_class: the number of cluster per class
        X, y = make_classification(
            n_samples=500, n_features=5, n_redundant=0, n_informative=4, random_state=42, n_clusters_per_class=1
        )
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)
        datasets = [
            make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable,
        ]
        X = datasets[0][0]
        y = datasets[0][1]

    elif name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target

    elif name == 'wine':
        df = pd.read_csv(".\\experiment_data\\dataset\\wine\\wine.data", sep=",", header=None)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]

    elif name == 'digits':
        from sklearn.datasets import load_digits
        digits = load_digits()
        n_samples = len(digits.images)
        X = digits.images.reshape((n_samples, -1))
        y = digits.target

    elif name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
    elif name == 'eeg_eye_state':
        df = pd.read_csv(os.getcwd() + "\\run_experiment\\experiment_data\\dataset\\eeg_eye_state\\eeg-eye-state_csv.csv"
                         , sep=",", header=None).drop(0)
        # print(os.getcwd() + "\\run_experiment\\experiment_data\\dataset\\eeg_eye_state\\eeg-eye-state_csv.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    elif name == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = data.data
        y = data.target

    elif name == 'glass':
        df = pd.read_csv("D:/Jupyter_Projs/W/MTBO/run_experiment/experiment_data/dataset/glass/glass.csv", )
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    else:
        raise NameError

    return X, y

def get_next_task(cr=1,  num_task=4, data=None, selected_tasks=None, gp_model=None,
                  design_x=None, betat=None, betap=None, iteration=0, cr4_seed=0):
    assert cr < len(betat)
    design_list = []
    tasks_mu = np.zeros(shape=[num_task, design_x.shape[0], 1])
    tasks_var = np.zeros(shape=[num_task, design_x.shape[0], 1])


    for t in range(num_task):
        design_list.append(np.hstack((design_x, t*np.ones(shape=[design_x.shape[0], 1]))))
    # 1.select task
    if cr == 0:
        tasks_acq_value = np.zeros(shape=[num_task])
        for t in range(num_task):
            count_task = np.count_nonzero(selected_tasks == t) + 1
            tasks_acq_value[t] = data[t]['Y'].max() + betat[cr]*np.power(count_task, -1/2) # 0.87, 1
            # print("betat[cr] = ", betat[cr] , "tasks_acq_value.max=", tasks_acq_value.max())
        next_task = tasks_acq_value.argmax()
        tasks_mu[next_task], tasks_var[next_task] = gp_model.predict(design_list[next_task],
                    Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})

        # print(f"tasks_var[next_task].max(){tasks_var[next_task].max()},  tasks_mu[next_task]{tasks_mu[next_task].max()}")
        # print("cr=", cr, ",   ", tasks_mu[next_task].max().round(5), np.power(count_task, -1/2).max())
        x_id = (tasks_mu[next_task] + betap[cr] * np.sqrt(tasks_var[next_task])).argmax()
        next_x = design_x[x_id]

    if cr == 1:
        for t in range(num_task):
            tasks_mu[t], tasks_var[t] = gp_model.predict(design_list[t],
                    Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})
        next_task = (tasks_mu + betat[cr] * np.sqrt(tasks_var)).max(axis=1).argmax()
        # next_task = gp.ICM.B.W.values.argmax()
        # print("cr=", cr, ",   ", tasks_mu.max().round(5), np.sqrt(tasks_var))
        x_id = (tasks_mu[next_task] + betap[cr] * np.sqrt(tasks_var[next_task])).argmax() # 0.3, 0.7
        next_x = design_x[x_id]

    if cr == 2:
        ave = 0.5*budget/num_task
        if iteration < 0.5*budget: # iteration has to be < , not <=
            next_task = int(iteration/ave)

        else:
            max_y_tasks = np.zeros(num_task)
            for t in range(num_task):
                max_y_tasks[t] = data[t]['Y'].max()
            next_task = max_y_tasks.argmax()

        task_mu, task_var = gp_model.predict(design_list[next_task],  # store single task variable, vector
                            Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})
        x_id = (tasks_mu + betap[cr] * np.sqrt(task_var)).argmax()
        next_x = design_x[x_id]

    if cr == 3:
        random.seed(cr4_seed)
        next_task = random.choice(range(num_task))
        task_mu, task_var = gp_model.predict(design_list[next_task],  # store single task variable, vector
                            Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})
        x_id = (tasks_mu + betap[cr] * np.sqrt(task_var)).argmax()
        next_x = design_x[x_id]

    return next_task, next_x, x_id


obj_name = 'ksvc'
bounds = [[1.e-05, 1.e+04], [1.e-04, 1.e-1]]
inputDim = len(bounds)
initN = 10
num_design = 2000
num_task = 3
v = 50
betat=[2, 2, 0, 0]
betap=[3, 2, 2, 2]
cr_list = [0,1,2,3] # [0,1,2,3] # choose criterion [0,1,2]
trials = 30
budget = 50
data_name = 'glass'  #['generate_classification', 'breast_cancer', 'wine', 'digits', 'iris', 'eeg_eye_state']
save_flag = False
save_flag2 = False
cwd_data = os.path.abspath(os.getcwd()) + "\\run_experiment\\experiment_data\\"  # 'D:\\Jupyter_Projs\\W\\MTBO\\' + "~~~'
init_path = os.path.join(cwd_data + f'init_data\\{obj_name}\\{data_name}', '')
design_path = os.path.join(cwd_data + f'design_data\\{obj_name}\\{data_name}', '')
result_path = os.path.join(cwd_data + f'result_data\\{obj_name}\\{data_name}', '')
for p in [init_path, design_path, result_path]:
    if not os.path.exists(p):
        os.makedirs(p)

cmt = GetCmtData(tasks_list=[3, 5, 7], dataset=select_dataset(data_name))
K = GPy.kern.Matern32(inputDim)
K.lengthscale.constrain_bounded(0.01, 1000)
# K.lengthscale.constrain_fixed()
# K =GPy.kern.RBF(inputDim)
K.variance.constrain_fixed(1)
#     km52 = GPy.kern.Matern52(input_dim=len(bounds), lengthscale=default_lengthscale,
#                              active_dims=list(range(len(bounds))), ARD=False)
#     km52.unlink_parameter(km52.variance)  # optimize 2 parameters: lengthscale, lik_variance
# K.unlink_parameter((K.variance))
icm = GPy.util.multioutput.ICM(input_dim=inputDim, num_outputs=num_task, kernel=K)
all_opt_results = {}
select_dict = {}
points_dict = {}
x_id_dict = {} # [cr, task, trial, budget]  准则0任务1在第2个trial中加入的x的id x_id_dict[0][]
# B_list = []
# B_list_max_index = []

for cr in cr_list:
    cr_opt_result = np.zeros(shape=(num_task, trials, budget))
    cr_x_id_result = -1*np.ones(shape=(num_task, trials, budget))
    cr_select_result = -1 * np.ones(shape=(trials, budget))
    cr_points_list = []
    for trial in range(trials):
        data = cmt.get_init_data(initN, bounds=bounds, save_path=init_path, seed=trial)
        design_x = get_design_domain(bounds, num_sampling=num_design, save_path=design_path, seed=trial)

        for iteration in range(budget):
            print(f"criterion:{cr}, trial:{trial+1}/{trials}, iteration:{iteration+1}/{budget}")
            # gp = GPy.models.GPCoregionalizedRegression([data[0]['X'], data[1]['X'], data[2]['X'], data[3]['X']],
            #                             [data[0]['Y'], data[1]['Y'], data[2]['Y'], data[3]['Y']], kernel=icm)
            gp = GPy.models.GPCoregionalizedRegression([data[i]['X'] for i in range(num_task)],
                                                       [data[i]['Y'] for i in range(num_task)], kernel=icm)
            # gp['.*rbf'], gp['.*Mat32.var'].constrain_fixed(1.) fix the sigma_f
            gp.mixed_noise.constrain_fixed(0.0001)
            gp.optimize()
            # B_list.append(gp.ICM.B.W.values)
            # B_list_max_index.append(gp.ICM.B.W.values.argmax())
            # print(gp.ICM.B.W)
            next_task, next_x, x_id = get_next_task(cr=cr, num_task=num_task, data=data,
                    selected_tasks=cr_select_result[trial, :iteration+1], gp_model=gp, design_x=design_x,
                    betat=betat, betap=betap, iteration=iteration, cr4_seed=trial+iteration*2)
            # next_task, next_x, x_id = get_ben_ave
            next_y = cmt.get_task_output(task_index=next_task, x=next_x)
            data[next_task]['X'] = np.vstack((data[next_task]['X'], next_x))
            data[next_task]['Y'] = np.vstack((data[next_task]['Y'], next_y))
            cr_select_result[trial, iteration] = next_task
            cr_x_id_result[next_task, trial, iteration] = x_id
            for t in range(num_task):
                cr_opt_result[t, trial, iteration] = data[t]['Y'].max()
        # print(f"trial:{trial+1}/{trials},", "the selected tasks are", str(cr_select_result[trial, :]))

        cr_points_list.append(data)

    init_select = np.tile(np.arange(num_task), (trials, 1)) # for np.unique, at least one choice of each task
    all_select = np.hstack([init_select, cr_select_result])
    select_dict[cr] = all_select
    points_dict[cr] = cr_points_list
    all_opt_results[cr] = cr_opt_result
    x_id_dict[cr] = cr_x_id_result

for cr, cr_opt_result in all_opt_results.items():
    for t in range(num_task):
        task_count_select = np.unique(select_dict[cr], return_counts=True)[1][t] - trials # delete init_select
        print(f"cr{cr}_t{t}___select={task_count_select}___ratio={round(task_count_select/(trials*budget), 4)}")


fig, ax = plt.subplots(figsize=(20, 12), tight_layout=True)
x_all_ticks = np.arange(1, budget+1, step=1)
marker_size = 15
y_font_size = 60
x_font_size = 60
tick_size = 40
legend_size = 60
colors_task = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#17becf', '#7f7f7f', '#bcbd22', '#e377c2']
markers_af = ["^", "s", "o", "*",  "v", "d", "x", "p", "+", "h"]
linestyle =  ['solid', 'dotted', 'dashed', 'dashdot']

for cr, cr_opt_result in all_opt_results.items():
    for t in range(num_task):
        m = cr_opt_result[t].mean(axis=0)
        s = cr_opt_result[t].std(axis=0)
        task_count_select = np.unique(select_dict[cr], return_counts=True)[1][t] - trials # delete init_select
        ax.plot(x_all_ticks, m, linewidth=3, label=f"c{cr}_t{t}_s{task_count_select}"
                , color=colors_task[t],  linestyle=linestyle[cr], markevery=5)
        ax.fill_between(x_all_ticks, m-0.05*s, m+0.05*s, color=colors_task[t], alpha=0.2)

ax.set_xlabel("Function evaluations", fontsize=x_font_size)
ax.set_ylabel("Accuracy", fontsize=y_font_size)
plt.tick_params(labelsize=tick_size) #设置坐标轴刻度字体大小
plt.legend(prop={'size': 30})
title_name = f"{obj_name}_trials={trials}, budget={budget}, criterion={cr_list},betat={betat}, betap={betap}"
plt.title(title_name, fontsize=15)
filename = result_path + title_name + ".pdf"
# plt.show()
if save_flag:
    # save the variable
    save_list = [all_opt_results, x_id_dict, points_dict, select_dict]
    output = open(result_path+title_name, 'wb')
    pickle.dump(save_list, output)
    output.close()
    # 如果同名pdf文件都存在， 在文件名字后面不断加a后，再保存
    if os.path.exists(filename):
        # save tasks' result
        filename = filename[:-4] + '_a' +'.pdf'
        plt.savefig(filename, dpi=300)
    else:
        # 如果不同名， 直接保存
        plt.savefig(filename, dpi=300)
    print("the file is saved in: ", result_path)


cr_colors = ['blue', 'red', 'green', 'purple']
cr_name = ["CMTBO-C1", "CMTBO-C2", "S-MTBO", "R-MTBO"]
fig, ax2 = plt.subplots(figsize=(18, 15), tight_layout=True)
# for cr in range(len(cr_list)):
for cr in cr_list:
    m = all_opt_results[cr].max(axis=0).mean(axis=0)  # 对某次iteration， 先求各个task中最大值，再求trial平均值
    s = all_opt_results[cr].max(axis=0).std(axis=0)
    ax2.plot(x_all_ticks, m, c=cr_colors[cr], label=cr_name[cr])
    ax2.fill_between(x_all_ticks, m - 0.05 * s, m + 0.05 * s, color=cr_colors[cr], alpha=0.2)

ax2.set_xlabel("Function evaluations", fontsize=x_font_size)
# ax2.set_ylabel("Best function value", fontsize=y_font_size)
ax2.set_ylabel("Accuracy", fontsize=y_font_size)
plt.tick_params(labelsize=tick_size)  # 设置坐标轴刻度字体大小
plt.legend(prop={'size': legend_size})
title_name = f" criterion={cr_list},betat={betat}, betap={betap}"
plt.title(title_name, fontsize=15)
cr_file_name = result_path + f"{obj_name}_criterion={cr_list},betat={betat}, betap={betap}" + ".pdf"
# plt.show()
if save_flag2:
    # 如果同名pdf文件都存在， 在文件名字后面不断加a后，再保存
    if os.path.exists(filename):
        # save criteria's result
        cr_file_name = cr_file_name[:-4] + '_a' + '.pdf'
        plt.savefig(cr_file_name, dpi=300)
    else:
        # 如果不同名， 直接保存
        plt.savefig(cr_file_name, dpi=300)
    print("the file is saved in: ", result_path)

plt.show()
