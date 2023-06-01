import random

import numpy as np
import pandas as pd
import scipy.stats as sci


def rename_keys(d, keys):
    return dict([(keys.get(k), v) for k, v in d.items()])


class OrdinalEncoderDistr:
    def __init__(self, obj_func='', C=None, lower_bound=[], upper_bound=[], ran_seed=None, shuffle=False,
                 randomized=False, sigma=0.01, bayesian=False, target=False, first=False):
        self.obj_func = obj_func
        self.C = C
        self.lb = lower_bound
        self.ub = upper_bound
        self.randomized = randomized
        self.sigma = sigma
        self.shuffle = shuffle
        self.ran_seed = ran_seed
        self.bayesian = bayesian
        self.first = first
        self.mapping = {}
        self.interval_mapping = {}
        self.step = None
        self.target = target
        self.col_noise = []

    def fit(self, X, y=None):
        y = y
        # Oridnal encoding
        for col in range(len(self.C)):
            choices = pd.DataFrame(X)[col].unique().tolist()
            encoded_value = range(0, len(choices))  # start from 0
            map_col = pd.Series(index=choices, data=encoded_value)
            self.mapping[col] = map_col
            # Mapping to the median of the uniform distribution
            if self.bayesian:
                # Compute the median of the uniform distribution
                self.step = (self.C[col] - 1) / self.C[col]  # C start form 0
                median_ = np.arange(start=self.step / 2, stop=self.C[col] - 1, step=self.step)
                self.mapping[col].iloc[:] = median_

            if self.target:
                # sort the index of current column by the target mean
                m_index = pd.DataFrame(y.reshape(-1)).groupby(pd.DataFrame(X)[col]).agg(['mean'])[0]. \
                    sort_values(by='mean', ascending=True).index
                # assign the sorted index to mapping,
                # so that the choices with a larger mean corresponds to a larger encoding distribution
                self.mapping[col].index = m_index
        return self

    def transform(self, data_array, shuffle=False, translation=None) -> object:
        df = pd.DataFrame(data_array)
        h_encoded = pd.DataFrame(data=np.zeros(shape=(df.shape[0], len(self.C))))
        x_continuous = df.iloc[:, len(self.C):]
        for col in range(len(self.C)):
            # shuflle should only be used in the iterations other than the first one.
            if shuffle:
                # noise = (encoded_data[col] - df[col].map(self.mapping[col])).values

                # shuffle the mapping of the encoding value
                col_value = self.mapping[col].values
                # self.mapping[col].iloc[:] = np.random.RandomState(seed=self.ran_seed + col).permutation(col_value)
                self.mapping[col].iloc[:] = np.random.permutation(col_value)

            h_encoded[col] = df[col].map(self.mapping[col])
            # adding noise as a uniform distribution
            if self.bayesian:
                if shuffle:
                    noise = translation[:, col]
                else:
                    noise = np.random.uniform(-1 * self.step * 1 / 2, self.step * 1 / 2, [df.shape[0]])
                    self.col_noise.append(noise)

                h_encoded[col] = h_encoded[col] + noise

        self.col_noise = np.array(self.col_noise).T
        z_enc = pd.concat([h_encoded, x_continuous], axis=1)

        # h_encoded[col] = data_array[col].map(self.mapping[col])

        # if self.first:
        #     d_boundry = pd.DataFrame(np.vstack(
        #         [data_array[np.where(data_array[:24, col] == 0)], data_array[np.where(data_array[:24, col] == 2)]]))
        #     y_boundry = np.vstack([y[np.where(data_array[:24, col] == 0)], y[np.where(data_array[:24, col] == 2)]])
        #     z_enc = pd.concat([z_enc, d_boundry], axis=0).reset_index(drop=True)
        #     y_enc = np.vstack([y, y_boundry])
        #     return z_enc, y_enc

        return z_enc


class UniformEncoder:

    def __init__(self, obj_func='', C=None, lower_bound=None, upper_bound=None, ran_seed=None, shuffle=False,
                 randomized=False, sigma=0.01, bayesian=False, first=False):
        if upper_bound is None:
            upper_bound = []
        if lower_bound is None:
            lower_bound = []
        self.obj_func = obj_func
        self.C = C
        self.lb = lower_bound
        self.ub = upper_bound
        self.randomized = randomized
        self.sigma = sigma
        self.mapping = {}
        self.shuffle = shuffle
        self.ran_seed = ran_seed
        self.bayesian = bayesian
        self.first = first
        self.ori_mapping = {}

    def fit(self, X, y):
        self.y = y

        for col in range(len(self.C)):
            num_c = self.C[col]
            coordinates_df = pd.Series(np.linspace(self.lb[col], self.ub[col], num_c, endpoint=False) + \
                                       (self.ub[col] - self.lb[col]) / (2 * num_c))
            self.mapping[col] = coordinates_df

        self.ori_mapping = self.mapping

        return self

    def transform(self, data_array) -> object:
        y = self.y
        df = pd.DataFrame(data_array)
        h_encoded = pd.DataFrame(data=np.zeros(shape=(df.shape[0], len(self.C))))
        x_continuous = df.iloc[:, len(self.C):]
        for col in range(len(self.C)):
            if self.shuffle:
                # shuffle the mapping of the encoding value
                col_value = self.mapping[col].values
                self.mapping[col].iloc[:] = np.random.RandomState(seed=self.ran_seed + col).permutation(col_value)
                print("\n mapping", self.mapping)
                # normal distribution
                # if self.bayesian:
                #     mu_arr = self.mapping[col].values
                #     sig = 0.1
                #     self.mapping[col].iloc[:] = np.random.normal(mu_arr, sig, [len(mu_arr)])
                #     -------check if the value in mapping is out of bound in af_bounds-------
            if self.bayesian:
                # adding noise as a uniform distribution
                h_encoded[col] = df[col].map(self.mapping[col])
                step = (self.ub[col] - self.lb[col]) / self.C[col]
                noise = np.random.uniform(-1 * step * 1 / 2, step * 1 / 2, [h_encoded.shape[0]])
                h_encoded[col] = h_encoded[col] + noise
                # self.mapping[col].iloc[:] = noise + self.mapping[col].iloc[:]
            z_enc = pd.concat([h_encoded, x_continuous], axis=1)
            # h_encoded[col] = data_array[col].map(self.mapping[col])
            if self.first:
                d_boundry = pd.DataFrame(np.vstack(
                    [data_array[np.where(data_array[:24, col] == 0)], data_array[np.where(data_array[:24, col] == 2)]]))
                y_boundry = np.vstack([y[np.where(data_array[:24, col] == 0)], y[np.where(data_array[:24, col] == 2)]])
                z_enc = pd.concat([z_enc, d_boundry], axis=0).reset_index(drop=True)
                y_enc = np.vstack([y, y_boundry])
                return z_enc, y_enc

        return z_enc


class TargetOrinidalEncoder:
    def __init__(self, obj_func='', C=None, lower_bound=[], upper_bound=[], ran_seed=None, shuffle=False,
                 randomized=False, sigma=0.01, bayesian=False, first=False):
        self.obj_func = obj_func
        self.C = C
        self.lb = lower_bound
        self.ub = upper_bound
        self.randomized = randomized
        self.sigma = sigma
        self.mapping = {}
        self.shuffle = shuffle
        self.ran_seed = ran_seed
        self.bayesian = bayesian
        self.first = first
        self.ori_mapping = {}

    # pd.DataFrame(y.reshape(-1)).groupby(pd.DataFrame(X)[1]).agg(['mean']).iloc[:, -1]
    def fit(self, X, y):
        self.y = y

        for col in range(len(self.C)):
            num_c = self.C[col]
            coordinates_df = pd.Series(np.linspace(self.lb[col], self.ub[col], num_c, endpoint=False) + \
                                       (self.ub[col] - self.lb[col]) / (2 * num_c))
            self.mapping[col] = coordinates_df

        self.ori_mapping = self.mapping

        return self

    def transform(self, data_array) -> object:
        y = self.y
        df = pd.DataFrame(data_array)
        h_encoded = pd.DataFrame(data=np.zeros(shape=(df.shape[0], len(self.C))))
        x_continuous = df.iloc[:, len(self.C):]
        for col in range(len(self.C)):
            if self.shuffle:
                # shuffle the mapping of the encoding value
                col_value = self.mapping[col].values
                self.mapping[col].iloc[:] = np.random.RandomState(seed=self.ran_seed + col).permutation(col_value)
                print("\n mapping", self.mapping)

            if self.bayesian:
                # adding noise as a uniform distribution
                h_encoded[col] = df[col].map(self.mapping[col])
                step = (self.ub[col] - self.lb[col]) / self.C[col]
                noise = np.random.uniform(-1 * step * 1 / 2, step * 1 / 2, [h_encoded.shape[0]])
                h_encoded[col] = h_encoded[col] + noise
                # self.mapping[col].iloc[:] = noise + self.mapping[col].iloc[:]
            z_enc = pd.concat([h_encoded, x_continuous], axis=1)
            # h_encoded[col] = data_array[col].map(self.mapping[col])
            if self.first:
                d_boundry = pd.DataFrame(np.vstack(
                    [data_array[np.where(data_array[:24, col] == 0)], data_array[np.where(data_array[:24, col] == 2)]]))
                y_boundry = np.vstack([y[np.where(data_array[:24, col] == 0)], y[np.where(data_array[:24, col] == 2)]])
                z_enc = pd.concat([z_enc, d_boundry], axis=0).reset_index(drop=True)
                y_enc = np.vstack([y, y_boundry])
                return z_enc, y_enc

        return z_enc


class stats_encoder:

    def __init__(self, cols=None, quantiles=[0.1, 0.9], mean_flag=False, randomized=True, sigma=0.01):
        self.cols = cols
        self.quantiles = quantiles
        self.randomized = randomized
        self.mean_flag = mean_flag
        self.sigma = sigma
        self.mapping0 = {}
        self.mapping1 = {}

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y.numpy().reshape(-1))
        # generate mapping
        for col in self.cols:

            stats0 = y.groupby(X[col]).agg([lambda X: np.quantile(X, self.quantiles[0])]).iloc[:, -1]

            if self.mean_flag:
                stats1 = y.groupby(X[col]).agg(['mean']).iloc[:, -1]
            else:
                stats1 = y.groupby(X[col]).agg([lambda X: np.quantile(X, self.quantiles[1])]).iloc[:, -1]

            self.mapping0[col] = stats0
            self.mapping1[col] = stats1

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        col_encoded = len(self.cols) * 2
        X_encoded = pd.DataFrame(data=np.zeros(shape=(X.shape[0], col_encoded)))

        for col in self.cols:
            X_encoded[col * 2] = X[col].map(self.mapping0[col])
            X_encoded[col * 2 + 1] = X[col].map(self.mapping1[col])

            if self.randomized:
                X_encoded[col * 2] = (X_encoded[col * 2] + np.random.normal(0, self.sigma, X_encoded.shape[0]))
                X_encoded[col * 2 + 1] = (X_encoded[col * 2 + 1] + np.random.normal(0, self.sigma, X_encoded.shape[0]))

        X_continuous = X.iloc[:, len(self.cols):]
        X_continuous.columns = range(2 * len(self.cols), len(self.cols) + X.shape[1])
        X_encoded = pd.concat([X_encoded, X_continuous], axis=1)

        return X_encoded


class TargetBayesianEncoder:
    def __init__(self, obj_func='', C=None, bayesian=False, first=False):
        self.obj_func = obj_func
        self.C = C
        self.bayesian = bayesian
        self.first = first
        self.mapping = {}
        self.interval_mapping = {}
        self.step = None
        self.col_noise = []

    def fit(self, X, y=None):

        for col in range(len(self.C)):
            choices = pd.DataFrame(X)[col].unique().tolist()
            encoded_value = range(0, len(choices))  # start from 0
            map_col = pd.Series(index=choices, data=encoded_value)
            self.mapping[col] = map_col
            # Compute the median of the uniform distribution
            self.step = (self.C[col] - 1) / self.C[col]  # C start form 0
            median_ = np.arange(start=self.step / 2, stop=self.C[col] - 1, step=self.step)
            self.mapping[col].iloc[:] = median_
            # sort the index of current column by the target mean
            m_index = pd.DataFrame(y.reshape(-1)).groupby(pd.DataFrame(X)[col]).agg(['mean'])[0]. \
                sort_values(by='mean', ascending=True).index
            # assign the sorted index to mapping,
            # so that the choices with a larger mean corresponds to a larger encoding distribution
            self.mapping[col].index = m_index

        return self

    def transform(self, data_array, translation=None) -> object:
        df = pd.DataFrame(data_array)
        h_encoded = pd.DataFrame(data=np.zeros(shape=(df.shape[0], len(self.C))))
        x_continuous = df.iloc[:, len(self.C):]
        for col in range(len(self.C)):
            h_encoded[col] = df[col].map(self.mapping[col])
            if translation is None:
                noise = np.random.uniform(-1 * self.step * 1 / 2, self.step * 1 / 2, [df.shape[0]])
            else:
                # store the existed noise, and translate it to new encoded interval
                noise = translation[:, col]
            h_encoded[col] = h_encoded[col] + noise
            self.col_noise.append(noise)

        self.col_noise = np.array(self.col_noise).T
        z_enc = pd.concat([h_encoded, x_continuous], axis=1)

        return z_enc


class PermutationEncoder:
    # every choice is mapping to a uniform distribution
    # the uniform distribution is generated by the median + random_noise
    def __init__(self, obj_func='', C=None, bayesian=False, first=False):
        self.obj_func = obj_func
        self.C = C
        self.bayesian = bayesian
        self.first = first
        self.mapping = {}
        self.interval_mapping = {}
        self.step = None
        self.col_noise = []

    def get_perm_index_list(self, num_perms):
        from itertools import permutations
        all_index_list = []  # 2 dimension, col, choice
        for col in range(len(self.C)):
            curr_cate_index_list = []
            # curr_perm_index_list = []
            # for p in range(num_perms):
            if self.C[col] < 5:
                col_perms = list(permutations(range(self.C[col])))
                # the number of choice is too small. return sampling
                if self.C[col] < num_perms:
                    col_random_index_list = np.random.choice(range(len(col_perms)), size=num_perms, replace=False)
                else:
                    # No return sampling for num_perms permutations
                    col_random_index_list = np.random.choice(range(len(col_perms)), size=num_perms, replace=False)
                for i in col_random_index_list:
                    curr_cate_index_list.append(col_perms[i])
                # all_index_list.append(curr_cate_index_list)
            # as the number of permutation is too large, random sampling
            else:
                init_perm_list = list(range(self.C[col]))
                # shuffle the ordered sequence
                # for i in range(num_perms):
                while len(curr_cate_index_list) < num_perms:
                    import copy
                    # new_perm = list(random.shuffle(init_perm_list))
                    random.shuffle(init_perm_list)
                    new_perm = copy.copy(init_perm_list)
                    # check duplicates
                    if new_perm not in curr_cate_index_list:
                        curr_cate_index_list.append(new_perm)
                    # attention! while the current list is duplicate with the old, current i should be neglected.

            all_index_list.append(curr_cate_index_list)

        return all_index_list

    def fit(self, data, num_perms):
        # from itertools import permutations
        all_perm_list = self.get_perm_index_list(num_perms=num_perms)

        for col in range(len(self.C)):
            self.mapping[col] = {}
            # focus on 1 categorical dimension
            choices = list(range(self.C[col]))
            # Compute the median of the uniform distribution
            self.step = (self.C[col] - 1) / self.C[col]  # C start form 0
            # as the medians in a col are the same, change the indexes to map different permutations
            median_ = np.arange(start=self.step / 2, stop=self.C[col] - 1, step=self.step)
            for p, key_perm in zip(range(len(all_perm_list[col])), all_perm_list[col]):
                self.mapping[col][p] = pd.Series(index=key_perm, data=median_)

        return self

    def transform(self, data_array, seed=42, num_perms=6, add_noise=True) -> object:
        z_list = []
        df = pd.DataFrame(data_array)
        x_continuous = df.iloc[:, len(self.C):]

        for p in range(num_perms):
            h_encoded = pd.DataFrame(data=np.zeros(shape=(df.shape[0], len(self.C))))
            for col in range(len(self.C)):
                h_encoded[col] = df[col].map(self.mapping[col][p])
                # different col and permutation have different noise
                if add_noise is True:
                    cur_seed = seed + col * 10 + p * 100
                    # Make sure the noise is the same as the second time the data is encoded
                    np.random.seed(cur_seed)
                    noise = np.random.uniform(-1 * self.step * 1 / 2, self.step * 1 / 2, [df.shape[0]])
                    h_encoded[col] = h_encoded[col] + noise

            z_enc = pd.concat([h_encoded, x_continuous], axis=1)
            z_list.append(z_enc)

        return z_list

# for single column
class PermFittingEncoder:
    # every choice is mapping to a uniform distribution
    # the uniform distribution is generated by the median + random_noise
    def __init__(self, obj_func='', C=None, bayesian=False, first=False):
        self.obj_func = obj_func
        self.C = C
        self.bayesian = bayesian
        self.first = first
        self.mapping = {}
        self.interval_mapping = {}
        self.step = None
        self.col_noise = []

    def get_perm_index_list(self, num_perms):
        from itertools import permutations
        all_index_list = []  # 2 dimension, col, choice
        for col in range(len(self.C)):
            curr_cate_index_list = []
            col_perms = list(permutations(range(self.C[col])))
            # all_index_list.append(curr_cate_index_list)

        return col_perms

    def fit(self):
        from itertools import permutations
        all_perm_list = [list(permutations(range(self.C[0])))]

        for col in range(len(self.C)):
            self.mapping[col] = {}
            # focus on 1 categorical dimension
            choices = list(range(self.C[col]))
            # Compute the median of the uniform distribution
            self.step = (self.C[col] - 1) / self.C[col]  # C start form 0
            # as the medians in a col are the same, change the indexes to map different permutations
            median_ = np.arange(start=self.step / 2, stop=self.C[col] - 1, step=self.step)
            for p, key_perm in zip(range(len(all_perm_list[col])), all_perm_list[col]):
                self.mapping[col][p] = pd.Series(index=key_perm, data=median_)

        return self

    def transform(self, data_array, seed=42, num_perms=6, add_noise=True) -> object:
        z_list = []
        df = pd.DataFrame(data_array)
        x_continuous = df.iloc[:, len(self.C):]

        for p in range(num_perms):
            h_encoded = pd.DataFrame(data=np.zeros(shape=(df.shape[0], len(self.C))))
            for col in range(len(self.C)):
                h_encoded[col] = df[col].map(self.mapping[col][p])
                # different col and permutation have different noise
                if add_noise is True:
                    cur_seed = seed + col * 10 + p * 100
                    # Make sure the noise is the same as the second time the data is encoded
                    np.random.seed(cur_seed)
                    noise = np.random.uniform(-1 * self.step * 1 / 2, self.step * 1 / 2, [df.shape[0]])
                    h_encoded[col] = h_encoded[col] + noise

            z_enc = pd.concat([h_encoded, x_continuous], axis=1)
            z_list.append(z_enc)

        return z_list
