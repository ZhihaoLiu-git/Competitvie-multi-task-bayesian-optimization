import os
import numpy as np
np.seterr(divide='ignore')
import pandas as pd
import GPy
import random
import pickle
from sklearn.svm import SVC, SVR
from sklearn import metrics, datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from scipy.stats import qmc

class GetCmtData:
    # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    def __init__(self, tasks_list=["linear", "poly", "rbf"], dataset=None):
        self.tasks_list = tasks_list
        self.svc_list = []
        self.dataset = dataset
    def get_init_data(self, num_sampling, bounds=[[0, 1], [0, 1]], save_path=None, seed=0):
        assert save_path is not None
        # Verify that the saving folder exists, if not, then create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        all_data = []
        for i in range(len(self.tasks_list)):
            file_name = save_path + f"task{i}_init" + str(seed)
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
                # sampler = qmc.LatinHypercube(d=2, optimization='random-cd') #seed=
                # sample = sampler.random(n=num_sampling)

                # solbol
                sampler = qmc.Sobol(d=len(bounds), scramble=False, seed=seed)
                sample = sampler.random_base2(m=2) # here, num_sampling is 2^m
                Xinit = qmc.scale(sample, l_bounds, u_bounds)
                for j in range(num_sampling):
                    yinit[j] = self.get_task_output(task_index=i, x=Xinit[j])

                task_data = {i: i, 'X': Xinit, 'Y': yinit}
                all_data.append(task_data)

                with open(file_name, 'wb') as data_file:
                    pickle.dump(task_data, data_file)

        return all_data
    # def get_task_output(self, task_index=None, x=None):
    #     assert task_index is not None and x is not None
    #     X, y = self.dataset
    #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #     clf = MLPClassifier(solver="adam", activation='relu', max_iter=2000,
    #                         alpha=self.tasks_list[task_index], hidden_layer_sizes=[int(x[0]), int(x[1])])
    #     scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    #     return scores.mean()
    def get_task_output(self, task_index=None, x=None):
        assert task_index is not None and x is not None
        X, y = self.dataset
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        clf = SVC(kernel=self.tasks_list[task_index], C=x[0], gamma=x[1])
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
    if name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target

    elif name == 'wine': #".\\experiment_data\\dataset\\wine\\wine.data"
        df = pd.read_csv(cwd_data + f'dataset\\{name}\\wine.data', sep=",", header=None)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
    else:
        raise NameError

    return X, y

def get_next_task(cr=1,  num_task=4, data=None, selected_tasks=None, gp_model=None,
                  design_x=None, betat=[5, 2], betap=[2, 2], iteration=0):
    assert cr < len(betat)
    design_list = []
    tasks_mu = np.zeros(shape=[num_task, design_x.shape[0], 1])
    tasks_var = np.zeros(shape=[num_task, design_x.shape[0], 1])


    for t in range(num_task):
        design_list.append(np.hstack((design_x, t*np.ones(shape=[design_x.shape[0], 1]))))
    # 1.select task
    if cr == 0: # y_max
        tasks_acq_value = np.zeros(shape=[num_task])
        for t in range(num_task):
            count_task = np.count_nonzero(selected_tasks == t) + 1
            tasks_acq_value[t] = data[t]['Y'].max() + betat[cr]*np.power(count_task, -1/2) # 0.87, 1
            # print("betat[cr] = ", betat[cr] , "tasks_acq_value.max=", tasks_acq_value.max())
        next_task = tasks_acq_value.argmax()
        tasks_mu[next_task], tasks_var[next_task] = gp_model.predict(design_list[next_task],
                    Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})
        # print("cr=", cr, ",   ", tasks_mu[next_task].max().round(5), np.power(count_task, -1/2).max())
        x_id = (tasks_mu[next_task] + betap[cr] * np.sqrt(tasks_var[next_task])).argmax()
        next_x = design_x[x_id]

    if cr == 1: # ucb
        for t in range(num_task):
            tasks_mu[t], tasks_var[t] = gp_model.predict(design_list[t],
                    Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})
        next_task = (tasks_mu + betat[cr] * np.sqrt(tasks_var)).max(axis=1).argmax()
        # next_task = gp.ICM.B.W.values.argmax()
        # print("cr=", cr, ",   ", tasks_mu.max().round(5), np.sqrt(tasks_var))
        x_id = (tasks_mu[next_task] + betap[cr] * np.sqrt(tasks_var[next_task])).argmax() # 0.3, 0.7
        next_x = design_x[x_id]

    if cr == 2: # 0.8
        ave = 0.8*budget/num_task
        if iteration < 0.8*budget: # iteration has to be < , not <=
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

    if cr == 3: # random
        next_task = random.choice(range(num_task))
        task_mu, task_var = gp_model.predict(design_list[next_task],  # store single task variable, vector
                            Y_metadata={'output_index': np.zeros((num_design, 1)).astype(int)})
        x_id = (tasks_mu + betap[cr] * np.sqrt(task_var)).argmax()
        next_x = design_x[x_id]

    return next_task, next_x, x_id

obj_name = 'svc'
bounds = [[1.e-05, 1.e+04], [1.e-04, 1.e-1]]
inputDim = len(bounds)
num_task = 3
initN = 4
num_design = 2000
v = 100
betat=[1, v, 0]
betap=[v, v, v]
cr_list = [0, 1, 2] # choose criterion [0,1,2]
data_name = 'wine'  #[breast_cancer', 'wine']
trials = 10
budget = 20
save_flag = False

cwd_data = os.path.abspath(os.getcwd()) +  "\\run_experiment\\experiment_data\\"   # 'D:\\Jupyter_Projs\\W\\MTBO\\' + "~'
init_path = os.path.join(cwd_data + f'init_data\\{obj_name}\\{data_name}', '')
design_path = os.path.join(cwd_data + f'design_data\\{obj_name}\\{data_name}', '')
result_path = os.path.join(cwd_data + f'result_data\\{obj_name}\\{data_name}', '')
for p in [init_path, design_path, result_path]:
    if not os.path.exists(p):
        os.makedirs(p)

cmt = GetCmtData(tasks_list=['linear', 'poly', 'rbf'], dataset=select_dataset(data_name))
K = GPy.kern.Matern32(inputDim)
K.lengthscale.constrain_bounded(0.001, 1000)
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
x_id_dict = {}

for cr in cr_list:
    cr_opt_result = np.zeros(shape=(num_task, trials, budget))
    cr_x_id_result = -1*np.ones(shape=(num_task, trials, budget))
    cr_select_result = -1 * np.ones(shape=(trials, budget))
    # cr_points_list = [[] for i in range(trials)]
    cr_points_list = []
    for trial in range(trials):
        # trial = 2
        data = cmt.get_init_data(initN, bounds=bounds, save_path=init_path, seed=trial*2)
        # data = cmt.get_init_data(initN, bounds=bounds, save_path=init_path, seed=2)
        design_x = get_design_domain(bounds, num_sampling=num_design, save_path=design_path, seed=trial)

        for iteration in range(budget):
            print(f"criterion:{cr}, trial:{trial+1}/{trials}, iteration:{iteration+1}/{budget}")
            gp = GPy.models.GPCoregionalizedRegression([data[0]['X'], data[1]['X'], data[2]['X']],
                                        [data[0]['Y'], data[1]['Y'], data[2]['Y']], kernel=icm)
            # print(gp)
            # gp['.*rbf'], gp['.*Mat32.var'].constrain_fixed(1.) fix the sigma_f
            gp.mixed_noise.constrain_fixed(0)
            gp.optimize()
            # B_list.append(gp.ICM.B.W.values)
            # B_list_max_index.append(gp.ICM.B.W.values.argmax())
            # print(gp.ICM.B.W)
            next_task, next_x, x_id = get_next_task(cr=cr, num_task=num_task, data=data,
                    selected_tasks=cr_select_result[trial, :iteration+1], gp_model=gp, design_x=design_x,
                    betat=betat, betap=betap, iteration=iteration)
            next_y = cmt.get_task_output(task_index=next_task, x=next_x)
            data[next_task]['X'] = np.vstack((data[next_task]['X'], next_x))
            data[next_task]['Y'] = np.vstack((data[next_task]['Y'], next_y))
            cr_select_result[trial, iteration] = next_task
            cr_x_id_result[next_task, trial, iteration] = x_id
            for t in range(num_task):
                cr_opt_result[t, trial, iteration] = data[t]['Y'].max()

        cr_points_list.append(data)

    init_select = np.tile(np.arange(num_task), (trials, 1)) # for np.unique, at least one choice of each task
    all_select = np.hstack([init_select, cr_select_result])
    select_dict[cr] = all_select
    points_dict[cr] = cr_points_list
    all_opt_results[cr] = cr_opt_result
    x_id_dict[cr] = cr_x_id_result


fig, ax = plt.subplots(figsize=(20, 12), tight_layout=True)
x_all_ticks = np.arange(1, budget+1, step=1)
marker_size = 15
y_font_size = 6
x_font_size = 6
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
        ax.fill_between(x_all_ticks, m-0.2*s, m+0.2*s, color=colors_task[t], alpha=0.2)

ax.set_xlabel("Function evaluations", fontsize=x_font_size)
ax.set_ylabel("Accuracy", fontsize=y_font_size)
plt.legend(prop={'size': 20})
title_name = f"{data_name}_trials={trials}, budget={budget}, criterion={cr_list},betat={betat}, betap={betap}"
plt.title(title_name, fontsize=15)
filename = result_path + title_name + ".pdf"
if save_flag:
    save_list = [all_opt_results, x_id_dict, points_dict, select_dict]
    output = open(result_path + title_name, 'wb')
    pickle.dump(save_list, output)
    output.close()
    if os.path.exists(filename):
        filename = filename[:-4] + '_a' + '.pdf'
        plt.savefig(filename, dpi=300)
    else:
        plt.savefig(filename, dpi=300)

plt.show()
