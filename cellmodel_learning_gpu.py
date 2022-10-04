# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Imports

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import h5py
from pathlib import Path
import time
from tqdm import tqdm
import csv
import os
from tqdm import tqdm, trange
import pydot

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedKFold, cross_val_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Bidirectional
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras import activations
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# %matplotlib inline
plt.rcParams['font.size'] = 16
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.family'] = "sans-serif"

# +
print(f'tf.test.is_built_with_cuda(): {tf.test.is_built_with_cuda()}');

if tf.__version__[0] == '1':
    print(f'tf.config.experimental_list_devices(): {tf.config.experimental_list_devices()}');
else:
    print(f'tf.config.list_physical_devices("GPU"): {tf.config.list_physical_devices("GPU")}');
    
print(f'tf.test.gpu_device_name(): {tf.test.gpu_device_name()}');
# -

# # Settings

# +
# randst = np.random.randint(0,100)
randst = 94
np.random.seed(randst)
print(randst)

verbosity = 0


# -

# # Define functions

# +
def compute_scores(y_test, y_hat):
    mae = mean_absolute_error(y_test, y_hat)
    mre = mean_absolute_percentage_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    metrics = [mae, mre, r2]
    mae_per = mean_absolute_error(y_test, y_hat, multioutput='raw_values')
    mre_per = mean_absolute_percentage_error(y_test, y_hat, multioutput="raw_values")
    r2_per = r2_score(y_test, y_hat, multioutput='raw_values')
    metrics_per = [mae_per, mre_per, r2_per]
    return metrics, metrics_per

def print_my_results(results_mae, results_mae_per, results_r2, results_r2_per, results_mre=None, results_mre_per=None):
    
    print('MAE:')
    print('\t%.3f (%.3f) overall' % (np.mean(results_mae), np.std(results_mae)))
    for mae,std,idx in zip(np.mean(results_mae_per, axis=0), np.std(results_mae_per, axis=0), idx_params):
        print('\t%.3f (%.3f) %s' % (mae, std, labels[idx]) )
    if results_mre is not None:
        print('MRE:')
        print('\t%.3f (%.3f) overall' % (np.mean(results_mre), np.std(results_mre)))
        for mre,std,idx in zip(np.mean(results_mre_per, axis=0), np.std(results_mre_per, axis=0), idx_params):
            print('\t%.3f (%.3f) %s' % (mre, std, labels[idx]) )
    print('R2:')
    print('\t%.3f (%.3f) overall' %(np.mean(results_r2), np.std(results_r2)))
    for r2,std,idx in zip(np.mean(results_r2_per, axis=0), np.std(results_r2_per, axis=0), idx_params):
        print('\t%.3f (%.3f) %s' % (r2, std, labels[idx]) )
        


# -

# # Import and partition data

scalar_on = 1
cdt_on = 0
drug_train_on = 1
downsample_on = 1
N_train = 4000

# ## Import data and test case indices

# +
name = "n4130"
here = Path.cwd()
data_path = Path(here.joinpath(f"data/data_control_{name}.h5"))

with h5py.File(data_path, 'r') as f:
    Datasetnames=f.keys()
    print(*list(Datasetnames), sep = "\n")
    trace = f['trace'][:,:200,:] # select time 0-200
    t = f['time'][...]
    adj_factors = f['adjustment_factors'][...]
    cost_terms = f['cost_terms'][...]
    
if trace.shape[0] != adj_factors.shape[0]:
    print('Number of samples do not match for trace and adj_factors!')

N_def = trace.shape[0]
print("Number of traces (control):", N_def)
labels = ["g_Kr","g_CaL","lambda_B","g_NaCa","g_K1","J_SERCA_bar","lambda_diff","lambda_RyR","g_bCa","g_Na","g_NaL"]

# Separate test cases (n=50) from all cases
idx_all = list(np.arange(0,trace.shape[0]))
idx_test = list(np.loadtxt(here.joinpath("data/idx_key_p11_s100_n5000_ns50.txt"), dtype=int))
idx_train = list(set(idx_all) - set(idx_test))

if drug_train_on:
    data_path = Path(here.joinpath(f"data/data_drug-combo_blockX_{name}.h5"))
    with h5py.File(data_path, 'r') as f:
        trace = np.vstack((trace, f['trace'][:,:200,:])) # select time 0-200
        adj_factors = np.vstack((adj_factors, f['adjustment_factors'][...]))
        cost_terms = np.vstack((cost_terms, f['cost_terms'][...]))
    idx_all = list(np.arange(0,trace.shape[0]))
    a = np.loadtxt(here.joinpath("data/idx_key_p11_s100_n5000_ns50.txt"), dtype=int)
    idx_test = list(a)
    idx_exclude = list(np.concatenate((a, a+N_def))) # remove test and modified test from idx_train
    idx_train = list(set(idx_all) - set(idx_exclude))
    N_def = trace.shape[0]
    print("Number of traces (all):", N_def)
# -

# ## Partition data
# Test cases are pre-selected for out-of-sample testing

# +
if not cdt_on:
    print(f"{cdt_on=}")
    idx_params = [0,1,4,9,10]
    trace_train = trace[idx_train,:,:]
    trace_test = trace[idx_test,:,:]
    af_train = adj_factors[idx_train,:][:,idx_params]
    af_test = adj_factors[idx_test,:][:,idx_params]

    if downsample_on:
        print(f"{downsample_on=}")
        idx_down = np.random.choice(af_train.shape[0], N_train, replace=False)
        trace_train = trace_train[idx_down,:,:]
        af_train = af_train[idx_down,:]
        
elif cdt_on:
    print(f"{cdt_on=}")
    import transportBasedTransforms.cdt as CDT
    trace_cdt = np.zeros_like(trace)
    N=200
    I0= (1.0/N)*np.ones(N)
    cdt=CDT.CDT(template=I0)

    for i in range(trace.shape[0]):
        for j in range(trace.shape[2]):
            trace_cdt[i,:,j] = cdt.transform(trace[i,:,j])

    idx_params = [0,1,4,9,10]
    trace_train = trace_cdt[idx_train,:,:]
    trace_test = trace_cdt[idx_test,:,:]
    af_train = adj_factors[idx_train,:][:,idx_params]
    af_test = adj_factors[idx_test,:][:,idx_params]
    
    if downsample_on:
        print(f"{downsample_on=}")
        idx_down = np.random.choice(af_train.shape[0], N_train, replace=False)
        trace_train = trace_train[idx_down,:,:]
        af_train = af_train[idx_down,:]
        
print("Number of training cases:", af_train.shape[0])
# -

# ## Drug effect target data
# Import as 4 datasets: control, drug-Kr, drug-CaL, drug-combo
#
# No partitioning needed since training with control data

# +
trace_drug = []
af_drug = []

block = "block20"
names_data = ["control", f"drug-Kr_{block}", f"drug-CaL_{block}", f"drug-combo_{block}"]

here = Path.cwd()

for name in names_data:
    data_path = Path(here.joinpath(f"data/data_{name}_n50.h5"))
    with h5py.File(data_path, "r") as f:
        trace_drug.append(f["trace"][:,:200,:])
        af_drug.append(f["adjustment_factors"][:,:])

trace_drug = np.array(trace_drug)
af_drug = np.array(af_drug)

# +
if not cdt_on:
    trace_drug_test = np.copy(trace_drug)
    af_drug_test = af_drug[:,:,idx_params]

elif cdt_on:
    trace_drug_test = np.zeros_like(trace_drug)

    for k in range(trace_drug_test.shape[0]):
        for i in range(trace_drug_test.shape[1]):
            for j in range(trace_drug_test.shape[3]):
                trace_drug_test[k,i,:,j] = cdt.transform(trace_drug[k,i,:,j])

    af_drug_test = af_drug[:,:,idx_params]
# -

# ## Drug effect target data - mps-prelim

# +
here = Path.cwd()

data_path = Path(here.joinpath("data/data_drug-prelim2_n9.h5"))
with h5py.File(data_path, "r") as f:
    trace_drug = np.zeros((9,200,2))
    print(list(f.keys()))
    trace_drug[:,:,0] = f["V"]
    trace_drug[:,:,1] = f["Ca"]
    
trace_test = trace_drug
# -

# # Parameter tuning for kNN, RF, SVM

# ## k-Nearest Neighbors
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train
if scalar_on:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

print('X shape:',X.shape)
print('Feature shape:',y.shape)

# +
knn_grid = {"n_neighbors": np.arange(1, 20),
            "weights": ['uniform', 'distance'],
            "p": [1,2],
              }

knn_base = KNeighborsRegressor()
knn_gscv = GridSearchCV(estimator = knn_base, param_grid = knn_grid,
                        cv=5, verbose=2,
                        n_jobs=-1)
knn_gscv.fit(X, y)

print(knn_gscv.best_params_)
# -

# ## Random Forest
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train
scaler = StandardScaler()
X = scaler.fit_transform(X)

print('X shape:',X.shape)
print('Feature shape:',y.shape)

# +
# Number of trees in Random Forest
rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]
rf_n_estimators.append(1500)
rf_n_estimators.append(2000)

# Maximum number of levels in tree
rf_max_depth = [int(x) for x in np.linspace(5, 55, 6)]
rf_max_depth.append(None)

# Number of features to consider at every split
# rf_max_features = ['auto', 'sqrt', 'log2']
rf_max_features = ['log2']

# Criterion to split on
rf_criterion = ['absolute_error']

# Minimum number of samples required to split a node
rf_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]

# Minimum decrease in impurity required for split to happen
rf_min_impurity_decrease = [0.0, 0.05, 0.1]

# Method of selecting samples for training each tree
rf_bootstrap = [True, False]

# Create the grid
rf_grid = {'n_estimators': rf_n_estimators,
               'max_depth': rf_max_depth,
               'max_features': rf_max_features,
               'criterion': rf_criterion,
               'min_samples_split': rf_min_samples_split,
               'min_impurity_decrease': rf_min_impurity_decrease,
               'bootstrap': rf_bootstrap}

rf_base = RandomForestRegressor()

# Create the random search Random Forest
rf_gscv = RandomizedSearchCV(estimator = rf_base, param_distributions = rf_grid,
                               n_iter=2, cv = 5, verbose = 2,
                               n_jobs = -1)

# Fit the random search model
rf_gscv.fit(X, y)

# View the best parameters from the random search
print(rf_gscv.best_params_)
with open("rfr_cv_output.txt","w") as f:
    f.write(rf_gscv.best_params_)

# -

# ## Support Vector
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train
if scalar_on:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

print('X shape:',X.shape)
print('Feature shape:',y.shape)

# +
gamma_range = list(np.logspace(-6, 3, 9))
gamma_range.append("scale")
gamma_range.append("auto")


svr_grid = {
#     "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "estimator__kernel": ["rbf"],
    "estimator__gamma": ["scale", "auto"],
    "estimator__C": [0.1, 1, 10, 100],
    "estimator__epsilon": [0.001, 0.01, 0.1, 1, 10],
#     "estimator__shrinking": [True, False]
}

start = time.time()
svr_base = MultiOutputRegressor(SVR())
svr_gscv = GridSearchCV(estimator = svr_base, param_grid = svr_grid,
                        cv=3, verbose=2,
                        n_jobs=-1)
svr_gscv.fit(X, y)
stop = time.time()
print(svr_gscv.best_params_)
print(stop-start)
# -

# ## Define best parameters for kNN, SVR

# +
# For normal data
# knn_best = {'n_neighbors': 7, 'p': 2, 'weights': 'distance'}
# svr_best = {'C': 1, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'} # actual from GridSearchCV
# svr_best = {'C': 0.01, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}

# For CDT data
knn_best = {'n_neighbors': 11, 'p': 2, 'weights': 'distance'}
svr_best = {'C': 100, 'epsilon': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
# -

# # Evaluate models using k-fold cross-validation
#
# https://scikit-learn.org/stable/modules/learning_curve.html
#

# ## k-Nearest Neighbor

# ### Prep data

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train

print('X shape:',X.shape)
print('Feature shape:',y.shape)


# -

# ### Define model

# +
def get_model_knn(**kwargs):
    model = KNeighborsRegressor(**kwargs,
                               n_jobs=-1)
    return model

def evaluate_model_knn(X, y):
    results_mae = []
    results_mae_per = []
    results_mre = []
    results_mre_per = []
    results_r2 = []
    results_r2_per = []
    
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=randst)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if scalar_on:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        # define model
        model = get_model_knn(**knn_best)
        # fit model
        model.fit(X_train, y_train)
        # evaluate model on test set
        y_hat = model.predict(X_test)
        
        scores, scores_per = compute_scores(y_test, y_hat)
        
        # store result
        print('>%.3f' % scores[0])
        results_mae.append(scores[0])
        results_mae_per.append(scores_per[0])
        results_mre.append(scores[1])
        results_mre_per.append(scores_per[1])
        results_r2.append(scores[2])
        results_r2_per.append(scores_per[2])
    return results_mae, results_mae_per, results_mre, results_mre_per, results_r2, results_r2_per



# -

# ### Run model with k-fold cross-validation

# +
start = time.time()
results_mae, results_mae_per, results_mre, results_mre_per, results_r2, results_r2_per = evaluate_model_knn(X, y)
stop = time.time()
print('Time of execution: %f' % (stop-start))

print_my_results(results_mae, results_mae_per, results_r2, results_r2_per)
# -

# ## Random Forest

# ### Prep data

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train

print('X shape:',X.shape)
print('Feature shape:',y.shape)


# -

# ### Define model using best_params

# +
def get_model_rfr(**kwargs):
    model = RandomForestRegressor(**kwargs)
    return model

def evaluate_model_rfr(X, y):
    results_mae = list()
    results_mae_per = list()
    results_r2 = list()
    results_r2_per = list()
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=randst)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if scalar_on:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        # define model
        model = get_model_rfr(**rf_gscv.best_params_)
        # fit model
        model.fit(X_train, y_train)
        # evaluate model on test set
        y_hat = model.predict(X_test)
        r2_per = r2_score(y_test,y_hat,multioutput='raw_values')
        r2 = r2_score(y_test,y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        mae_per = mean_absolute_error(y_test, y_hat, multioutput='raw_values')
        # store result
        print('>%.3f' % mae)
        results_mae.append(mae)
        results_mae_per.append(mae_per)
        results_r2.append(r2)
        results_r2_per.append(r2_per)
    return results_mae, results_mae_per, results_r2, results_r2_per


# -

# ### Run model with k-fold cross-validation

# +
start = time.time()
results_mae, results_mae_per, results_r2, results_r2_per = evaluate_model_rfr(X, y)
stop = time.time()
print('Time of execution: %f' % (np.divide(stop-start, 60)))

print_my_results(results_mae, results_mae_per, results_r2, results_r2_per)
# -

# ## Support Vector Regression

# ### Prep data

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train

print('X shape:',X.shape)
print('Feature shape:',y.shape)


# -

# ### Define model

# +
def get_model_svr(**kwargs):
    svr = SVR(**kwargs)

    model = MultiOutputRegressor(svr)
    return model

def evaluate_model_svr(X, y):
    results_mae = list()
    results_mae_per = list()
    results_r2 = list()
    results_r2_per = list()
    svr_train_error = list()
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=randst)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if scalar_on:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        # define model
        model = get_model_svr(**svr_best)
        # fit model
        model.fit(X_train, y_train)
        # evaluate model on test set
        y_hat = model.predict(X_test)
        scores, scores_per = compute_scores(y_test, y_hat)
        svr_train_error.append(mean_absolute_error(y_train, model.predict(X_train)))
        # store result
        print('>%.3f' % scores[0])
        results_mae.append(scores[0])
        results_mae_per.append(scores_per[0])
        results_r2.append(scores[2])
        results_r2_per.append(scores_per[2])
    return results_mae, results_mae_per, results_r2, results_r2_per, svr_train_error


# -

# ### Run model with k-fold cross-validation

# +
start = time.time()
results_mae, results_mae_per, results_r2, results_r2_per, svr_train_error = evaluate_model_svr(X, y)
stop = time.time()
print('Time of execution: %f' % (np.divide(stop-start, 60)))

print_my_results(results_mae, results_mae_per, results_r2, results_r2_per)
# -

# ### Evaluate ratio of training and validation error (last epoch)

# +
svr_validate = np.copy(results_mae)
results_svr = np.zeros((15,3))
for i in range(15):
    results_svr[i,0] = svr_train_error[i]
    results_svr[i,1] = svr_validate[i]
    results_svr[i,2] = svr_validate[i]/svr_train_error[i]
    
np.savetxt("train_test_loss_svr.csv", results_svr, delimiter=',')
# -

# ## Multi-Layer Perceptron

# ### Prep data

# +
X = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
y = af_train

print('X shape:',X.shape)
print('Feature shape:',y.shape)
# -

# ### Define model

# +
dropout = 0.0
weight_reg = 0.001
lr = 0.001
loss_metric = "mae"
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    
def get_model_mlp1(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(500, input_dim=n_inputs,
                    kernel_initializer='he_uniform',
                    activation=activations.swish,
                    kernel_regularizer=l2(weight_reg), bias_regularizer=l2(weight_reg)
                   )
             )
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation=activations.tanh))
    model.compile(loss='mae', optimizer=Adam(learning_rate=lr), metrics="mae")
    return model

def get_model_mlp3(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(100, input_dim=n_inputs,
                    kernel_initializer='he_uniform',
                    activation=activations.swish,
                    kernel_regularizer=l2(weight_reg), bias_regularizer=l2(weight_reg)
                   )
             )
    model.add(Dropout(dropout))
    model.add(Dense(100, activation=activations.swish,
                    kernel_regularizer=l2(weight_reg), bias_regularizer=l2(weight_reg)
                   )
             )
    model.add(Dropout(dropout))
    model.add(Dense(100, activation=activations.swish,
                    kernel_regularizer=l2(weight_reg), bias_regularizer=l2(weight_reg)
                   )
             )
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation=activations.tanh))
    model.compile(loss='mae', optimizer=Adam(learning_rate=lr), metrics="mae")
    return model

def evaluate_model_mlp(X, y):
    results_mae = list()
    results_mae_per = list()
    results_r2 = list()
    results_r2_per = list()
    mlp_trained = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=randst)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        if scalar_on:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        model = get_model_mlp1_test(n_inputs, n_outputs) # <<<<< SPECIFY WHICH MLP MODEL
        
        mlp_trained.append(model.fit(X_train, y_train, validation_split = 0.1, shuffle = False, epochs=200,
                                     batch_size=32,
                                     verbose=verbosity,
#                                      callbacks=[callback]
                                     )
                          )
        
        y_hat = model.predict(X_test)
        scores, scores_per = compute_scores(y_test, y_hat)
        # store result
        print('>%.3f' % scores[0])
        results_mae.append(scores[0])
        results_mae_per.append(scores_per[0])
        results_r2.append(scores[2])
        results_r2_per.append(scores_per[2])
    return results_mae, results_mae_per, results_r2, results_r2_per, mlp_trained


# -

# ### Run model with k-fold cross-validation

# +
start = time.time()
results_mae, results_mae_per, results_r2, results_r2_per, mlp_trained = evaluate_model_mlp(X, y)
stop = time.time()
print('Time of execution: %f' % (np.divide(stop-start, 60)))

print_my_results(results_mae, results_mae_per, results_r2, results_r2_per)
# -

name = "mlp_1layer100n_dropout0_regular0_batch32_cdt"

# ### Evaluate ratio of training and validation error (last epoch)

# +
mlp_validate = np.copy(results_mae)
results_mlp = np.zeros((15,3))
for i in range(15):
    results_mlp[i,0] = mlp_trained[i].history["mae"][-1]
    results_mlp[i,1] = mlp_validate[i]
    results_mlp[i,2] = mlp_validate[i]/mlp_trained[i].history["mae"][-1]
    
np.savetxt(f"overfit_test/history_{name}.csv", results_mlp, delimiter=',')
    
# -

# ### Plot learning curves

# +
fig, axs = plt.subplots(3,5,figsize=(14,8), sharey = True, sharex = True)
axs = axs.T.flatten()
for i, ax in enumerate(axs):
    ax.plot(mlp_trained[i].history["loss"])
    ax.plot(mlp_trained[i].history["val_loss"])
    ax.set_xticks([0,100,200])
    if i in [0,3,6,9,12,15]:
        ax.set_title(f"Fold {i//3+1:d}")
        
axs[-1].legend(["Train","Validation"], prop={'size': 12})
fig.supxlabel('Epoch')
fig.supylabel('MAE')

plt.savefig(f"overfit_test/learning_curve_{name}.png", dpi=300, bbox_inches="tight")
# -

# ## Convolutional Neural Network

# ### Prep data

# +
X = np.copy(trace_train)
y = af_train

print('X shape:',X.shape)
print('Feature shape:',y.shape)
# -

# ### Define model

# +
dropout = 0.0
weight_reg = 0.001
lr = 0.001
conv2d_on = True

def get_model_cnn3(n_inputs, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=6, input_shape=(200,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activations.swish))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=32, kernel_size=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activations.swish))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=64, kernel_size=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activations.swish))
    model.add(MaxPooling1D(pool_size=5, strides=2))
    model.add(Flatten())
    model.add(Dense(64, activation=activations.swish))
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation=activations.tanh))
    model.compile(loss='mae', optimizer='adam', metrics="mae")
    return model

def get_model_cnn1(n_inputs, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, input_shape=n_inputs))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation=activations.swish))
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation=activations.tanh))
    model.compile(loss='mae', optimizer='adam', metrics="mae")
    return model


def get_model_cnn1_2d(n_inputs, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=32,
                     kernel_size=2,
                     input_shape=(200, 2, 1),
                     kernel_regularizer=l2(weight_reg),
                     # bias_regularizer=l2(weight_reg),
                    )
             )
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation=activations.swish,
                    kernel_regularizer=l2(weight_reg),
                    # bias_regularizer=l2(weight_reg)
                   )
             )
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation=activations.tanh))
    model.compile(loss='mae', optimizer=Adam(learning_rate=lr), metrics="mae")
    return model


def evaluate_model_cnn(X, y):
    results_mae = list()
    results_mae_per = list()
    results_r2 = list()
    results_r2_per = list()
    cnn_trained = list()
    n_inputs, n_outputs = X.shape[1:], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=randst)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
        if scalar_on:
            scalers = {}
            for i in range(X_train.shape[2]):
                scalers[i] = StandardScaler()
                X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])
            for i in range(X_test.shape[2]):
                X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])
        if conv2d_on:
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)
        model = get_model_cnn1_2d(n_inputs, n_outputs)
        
        cnn_trained.append(model.fit(X_train, y_train, validation_split = 0.1, shuffle = False, epochs=200,
                                      verbose=verbosity
                                     )
                          )
        
        y_hat = model.predict(X_test)
        scores, scores_per = compute_scores(y_test, y_hat)
        # store result
        print('>%.3f' % scores[0])
        results_mae.append(scores[0])
        results_mae_per.append(scores_per[0])
        results_r2.append(scores[2])
        results_r2_per.append(scores_per[2])
    return results_mae, results_mae_per, results_r2, results_r2_per, cnn_trained


# -

# ### Run model with k-fold cross-validation

# +
start = time.time()
results_mae, results_mae_per, results_r2, results_r2_per, cnn_trained = evaluate_model_cnn(X, y)
stop = time.time()
print('Time of execution: %f' % (np.divide(stop-start, 60)))

print_my_results(results_mae, results_mae_per, results_r2, results_r2_per)
# -

name = "cnn1-2d"

# ### Evaluate ratio of training and validation error (last epoch)

# +
cnn_validate = np.copy(results_mae)
results_cnn = np.zeros((15,3))
for i in range(15):
    results_cnn[i,0] = cnn_trained[i].history["loss"][-1]
    results_cnn[i,1] = cnn_validate[i]
    results_cnn[i,2] = cnn_validate[i]/cnn_trained[i].history["loss"][-1]
    
# np.savetxt(f"overfit_test/history_{name}.csv", results_cnn, delimiter=',')
    
# -

# ### Plot learning curves

# +
fig, axs = plt.subplots(3,5,figsize=(14,8), sharey = True, sharex = True)
axs = axs.T.flatten()
for i, ax in enumerate(axs):
    ax.plot(cnn_trained[i].history["loss"])
    ax.plot(cnn_trained[i].history["val_loss"])
    ax.set_xticks([0,100,200])
    if i in [0,3,6,9,12,15]:
        ax.set_title(f"Fold {i//3+1:d}")
        
axs[-1].legend(["Train","Validation"], prop={'size': 12})
fig.supxlabel('Epoch')
fig.supylabel('MAE')

plt.savefig(f"overfit_test/learning_curve_{name}.png", dpi=300, bbox_inches="tight")
# -

# ## Fully Convolutional Neural Networks (FCN)

# ### Prep data

# +
X = np.copy(trace_train)
y = af_train

print('X shape:',X.shape)
print('Feature shape:',y.shape)


# -

# ### Define model

# +
def get_model_fcn(n_inputs, n_outputs):
    x = keras.layers.Input(n_inputs)
    drop_out = Dropout(0.1)(x)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=8, input_shape=n_inputs, padding='same')(x) # default filter 128
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activations.swish)(conv1)

    drop_out = Dropout(0.1)(conv1)
    conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv1) # default filter 256
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation(activations.swish)(conv2)

    drop_out = Dropout(0.1)(conv2)
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv2) # default filter 128
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation(activations.swish)(conv3)

    full = keras.layers.GlobalAveragePooling1D()(conv3)
#     full = keras.layers.GlobalMaxPooling1D()(conv3)
    out = keras.layers.Dense(n_outputs)(full)
    model = keras.models.Model(inputs=x, outputs=out)

    optimizer = keras.optimizers.Adam()
    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

def get_model_fcn_2(n_inputs, n_outputs):
    model = Sequential()
#     model.add(Dropout(0.1))
    model.add(Conv1D(filters=128, kernel_size=8, input_shape=n_inputs, padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    
#     model.add(Dropout(0.1))
    model.add(Conv1D(filters=256, kernel_size=5, input_shape=n_inputs, padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    
#     model.add(Dropout(0.1))
    model.add(Conv1D(filters=128, kernel_size=3, input_shape=n_inputs, padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    
    model.add(GlobalAveragePooling1D())
#     model.add(MaxPooling1D())
#     model.add(AveragePooling1D())
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    return model
    
def evaluate_model_fcn(X, y):
    results_mae = list()
    results_mae_per = list()
    results_r2 = list()
    results_r2_per = list()
    fcn_trained = list()
    n_inputs, n_outputs = X.shape[1:], y.shape[1]
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=randst)
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix,:,:], X[test_ix,:,:]
        y_train, y_test = y[train_ix], y[test_ix]
        nbatch, n_time, n_channel  = X_train.shape[0], X_train.shape[1], X_train.shape[2]
        if scalar_on:
            scalers = {}
            for i in range(X_train.shape[2]):
                scalers[i] = StandardScaler()
                X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])
            for i in range(X_test.shape[2]):
                X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])
        model = get_model_fcn(n_inputs, n_outputs)
        
        fcn_trained.append(model.fit(X_train, y_train,
                                     validation_split = 0.1,
                                     shuffle = True,
                                     epochs=300,
                                     verbose=verbosity
                                     )
                          )
        
        y_hat = model.predict(X_test)
        scores, scores_per = compute_scores(y_test, y_hat)
        # store result
        print('>%.3f' % scores[0])
        results_mae.append(scores[0])
        results_mae_per.append(scores_per[0])
        results_r2.append(scores[2])
        results_r2_per.append(scores_per[2])
    return results_mae, results_mae_per, results_r2, results_r2_per, fcn_trained



# -

# ### Run model with k-fold cross-validation

# +
start = time.time()
results_mae, results_mae_per, results_r2, results_r2_per, fcn_trained = evaluate_model_fcn(X, y)
keras.backend.clear_session()
stop = time.time()
print('Time of execution: %f' % (np.divide(stop-start, 60)))

print_my_results(results_mae, results_mae_per, results_r2, results_r2_per)
# -

# ### Evaluate ratio of training and validation error (last epoch)

# +
fcn_validate = np.copy(results_mae)
results_fcn = np.zeros((15,3))
for i in range(15):
    results_fcn[i,0] = fcn_trained[i].history["mae"][-1]
    results_fcn[i,1] = fcn_validate[i]
    results_fcn[i,2] = fcn_validate[i]/fcn_trained[i].history["mae"][-1]
    
np.savetxt("fcn_train_test_loss.csv", results_fcn)
# -

# ### Plot learning curves

# +
fig, axs = plt.subplots(3,5,figsize=(14,8), sharey = True, sharex = True)
axs = axs.T.flatten()
for i, ax in enumerate(axs):
    ax.plot(fcn_trained[i].history["loss"])
    ax.plot(fcn_trained[i].history["val_loss"])
    ax.set_xticks([0,100,200])
    if i in [0,3,6,9,12,15]:
        ax.set_title(f"Fold {i//3+1:d}")
        
axs[-1].legend(["Train","Validation"], prop={'size': 12})
fig.supxlabel('Epoch')
fig.supylabel('MAE')

plt.savefig("learning_curve_fcn_300epoch.png", dpi=300, bbox_inches="tight")


# -

# # Final train and evaluation on target test cases
#
# Note: need to run "Define model..." cells above to define model functions

# ## Prep data for model

# +
def prep_data_knn():
    X_train = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
    X_test = np.concatenate((trace_test[:,:,0],trace_test[:,:,1]),axis=1)
    y_train = af_train
    y_test = af_test

    if scalar_on:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prep_data_knn():
    X_train = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
    X_test = np.concatenate((trace_test[:,:,0],trace_test[:,:,1]),axis=1)
    y_train = af_train
    y_test = af_test

    if scalar_on:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prep_data_knn():
    X_train = np.concatenate((trace_train[:,:,0],trace_train[:,:,1]),axis=1)
    X_test = np.concatenate((trace_test[:,:,0],trace_test[:,:,1]),axis=1)
    y_train = af_train
    y_test = af_test

    if scalar_on:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def prep_data_mlp():
    
    X_train = np.concatenate((trace_train[:,:,0], trace_train[:,:,1]), axis=1)
    X_test = np.concatenate((trace_test[:,:,0], trace_test[:,:,1]), axis=1)
    y_train = af_train
    y_test = af_test

    if scalar_on:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print("Standard Scalar:", scalar_on)
    print("Training size:", y_train.shape[0])
    print("Drug training:", drug_train_on)
    print("Dropout:", dropout)
    print("Weight reg:", weight_reg)
    
    return X_train, X_test, y_train, y_test

def prep_data_cnn():
    X_train = np.copy(trace_train)
    X_test = np.copy(trace_test)
    y_train = af_train
    y_test = af_test

    if scalar_on:
        scalers = {}
        for i in range(X_train.shape[2]):
            scalers[i] = StandardScaler()
            X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])
        for i in range(X_test.shape[2]):
            X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])

    if conv2d_on:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

    print("Standard Scalar:", scalar_on)
    print("Training size:", y_train.shape[0])
    print("Drug training:", drug_train_on)
    print("Dropout:", dropout)
    print("Weight reg:", weight_reg)
    
    return X_train, X_test, y_train, y_test


# -

# ## kNN

# +
model = get_model_knn(**knn_best)

t_train_start = time.time()
model.fit(X_train, y_train)
t_train_stop = time.time()

t_test_start = time.time()
y_hat = model.predict(X_test).astype(float)
t_test_stop = time.time()

print("Time to train model: ", t_train_stop-t_train_start)
print("Time to test model: ", t_test_stop-t_test_start)

# +
y_drug_hat = np.zeros((4,50,5))

for i in range(af_drug_test.shape[0]):
    X_drug_test = np.concatenate((trace_drug_test[i,:,:,0],trace_drug_test[i,:,:,1]),axis=1)
    if scalar_on:
        X_drug_test = scaler.transform(X_drug_test)
    y_drug_hat[i,:,:] = model.predict(X_drug_test)
# -

# Control
with h5py.File("results_knn.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_hat, dtype='float')

# Control + drug
with h5py.File("results_drug_knn_cdt.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_drug_hat, dtype='float')

# ## Random Forest

# +
model = get_model_rfr()

t_train_start = time.time()
model.fit(X_train, y_train)
t_train_stop = time.time()

t_test_start = time.time()
y_hat = model.predict(X_test)
t_test_stop = time.time()

print("Time to train model: ", t_train_stop-t_train_start)
print("Time to test model: ", t_test_stop-t_test_start)
# -

# ## Support Vector

# +
model = get_model_svr(**svr_best)

t_train_start = time.time()
model.fit(X_train, y_train)
t_train_stop = time.time()

t_test_start = time.time()
y_hat = model.predict(X_test)
t_test_stop = time.time()

print("Time to train model: ", t_train_stop-t_train_start)
print("Time to test model: ", t_test_stop-t_test_start)

# +
y_drug_hat = np.zeros((4,50,5))

for i in range(af_drug_test.shape[0]):
    X_drug_test = np.concatenate((trace_drug_test[i,:,:,0],trace_drug_test[i,:,:,1]),axis=1)
    if scalar_on:
        X_drug_test = scaler.transform(X_drug_test)
    y_drug_hat[i,:,:] = model.predict(X_drug_test)
# -

# Control
with h5py.File("results_svr.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_hat, dtype='float')

# Control + drug
with h5py.File("results_drug_svr.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_drug_hat, dtype='float')

# ## MLP

# +
X_train, X_test, y_train, y_test = prep_data_mlp()
n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]

model = get_model_mlp3(n_inputs, n_outputs)
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=False, verbose=1)
callback = []

t_train_start = time.time()
model.fit(X_train, y_train, epochs=100, verbose=verbosity, callbacks=[callback])
t_train_stop = time.time()

t_test_start = time.time()
y_hat = model.predict(X_test)
t_test_stop = time.time()

print("Time to train model: ", t_train_stop-t_train_start)
print("Time to test model: ", t_test_stop-t_test_start)
# -

np.savetxt("results_mlp3_drug-train_mps-prelim2.csv", y_hat, delimiter=",")

# +
y_drug_test = af_drug_test
y_drug_hat = np.zeros((4,50,5))
experiment = ["control", "Kr", "CaL", "Kr+CaL"]

for i in range(af_drug_test.shape[0]):
    X_drug_test = np.concatenate((trace_drug_test[i,:,:,0], trace_drug_test[i,:,:,1]), axis=1)
    if scalar_on:
        X_drug_test = scaler.transform(X_drug_test)
    y_drug_hat[i,:,:] = model.predict(X_drug_test)
    print(experiment[i])
    print(r2_score(y_drug_test[i], y_drug_hat[i], multioutput='raw_values'))
    print(mean_absolute_error(y_drug_test[i], y_drug_hat[i], multioutput='raw_values'))
# -

# Control
with h5py.File("results_mlp3-drugtrain-03.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_hat, dtype='float')

# Control + drug
with h5py.File("results_drug_mlp3-drugtrain-03.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_drug_hat, dtype='float')

# ## CNN

# +
X_train, X_test, y_train, y_test = prep_data_cnn()
n_inputs, n_outputs = X_train.shape[1:], y_train.shape[1]

model = get_model_cnn1_2d(n_inputs, n_outputs) # <<< CHOOSE THIS
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=False, verbose=1)

t_train_start = time.time()
model.fit(X_train, y_train, epochs=100, verbose=verbosity, callbacks = [callback])
t_train_stop = time.time()

t_test_start = time.time()
y_hat = model.predict(X_test)
t_test_stop = time.time()

print("Time to train model: ", t_train_stop-t_train_start)
print("Time to test model: ", t_test_stop-t_test_start)
# -

np.savetxt("results_cnn1-2d_drugtrain_mlp-prelim2.csv", y_hat, delimiter=",")

# +
y_drug_test = af_drug_test
y_drug_hat = np.zeros((4,50,5))
experiment = ["control", "Kr", "CaL", "Kr+CaL"]

for i in range(af_drug_test.shape[0]):
    X_drug_test = np.copy(trace_drug_test[i,:,:,:])
    if scalar_on:
        for j in range(X_test.shape[2]):
            X_drug_test[:, :, j] = scalers[j].transform(X_drug_test[:, :, j])
    if conv2d_on:
        X_drug_test = np.expand_dims(X_drug_test, axis=-1)
    y_drug_hat[i,:,:] = model.predict(X_drug_test)
    print(experiment[i])
    print(r2_score(y_drug_test[i], y_drug_hat[i], multioutput='raw_values'))
    print(mean_absolute_error(y_drug_test[i], y_drug_hat[i], multioutput='raw_values'))
# -

# Control
with h5py.File("results_cnn1-2d-drugtrain.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_hat, dtype='float')

# Control + drug
with h5py.File("results_drug_cnn1-2d-drugtrain.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_drug_hat, dtype='float')

# ## FCN

# +
X_train = trace_train
X_test = trace_test
y_train = af_train
y_test = af_test

if scalar_on:
    scalers = {}
    for i in range(X_train.shape[2]):
        scalers[i] = StandardScaler()
        X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])
    for i in range(X_test.shape[2]):
        X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])

# +
n_inputs, n_outputs = X_train.shape[1:], y_train.shape[1]

model = get_model_fcn(n_inputs, n_outputs)

t_train_start = time.time()
model.fit(X_train, y_train, epochs=200, verbose=verbosity)
t_train_stop = time.time()

t_test_start = time.time()
y_hat = model.predict(X_test)
t_test_stop = time.time()

print("Time to train model: ", t_train_stop-t_train_start)
print("Time to test model: ", t_test_stop-t_test_start)
# -

with h5py.File("results_fcn.h5","w") as f:
    f.create_dataset("adjustment_factors", data=y_hat, dtype='float')

# ## Mean of $y_{train}$

# +
y_train = np.mean(af_train,axis=0)
y_test = af_test

mae_if_mean = mean_absolute_error(y_test, np.tile(y_train, (y_test.shape[0],1)), multioutput='raw_values')
print(mae_if_mean)
# -

# # Ensemble prediction interval

# ## Define functions

# +
tf.get_logger().setLevel('ERROR')

# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test, **kwargs_fit):
    ensemble = list()
    for i in tqdm(range(n_members)):
        n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]
#         model = get_model_mlp3(n_inputs, n_outputs)
        model = get_model_cnn1_2d(n_inputs, n_outputs)
        model.fit(X_train, y_train, **kwargs_fit)
        ensemble.append(model)
    return ensemble


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X_i):
    # make predictions
    y_hat_i = np.asarray([model.predict(X_i, verbose=0) for model in ensemble])[:,0,:]
    # mean value across ensemble
    mean_y_hat_i = np.mean(y_hat_i, axis=0)
    interval = np.std(y_hat_i, axis=0)
    lower_i = mean_y_hat_i - interval
    upper_i = mean_y_hat_i + interval
    return y_hat_i, mean_y_hat_i, lower_i, upper_i



# -

# ## Train ensemble of models

n_members = 500
X_train, X_test, y_train, y_test = prep_data_cnn()
ensemble = fit_ensemble(n_members, X_train, X_test, y_train, y_test, verbose=verbosity, epochs=100)

# ## Make prediction on test set with prediction interval

# +
saveit=1

# Compute prediction interval for each point in y_test
y_hat_all = np.zeros((50, 500, 5))
y_lower = np.zeros_like(y_test)
y_upper = np.zeros_like(y_test)
y_mean = np.zeros_like(y_test)
for i in tqdm(range(y_test.shape[0])):
    X_i = np.expand_dims(X_test[i, :], axis=0)
    y_hat_all[i,:,:], y_mean[i,:], y_lower[i,:],  y_upper[i,:] = predict_with_pi(ensemble, X_i)
    
if saveit:
    np.save("results_pi_cnn1-2d-02_ne500.npy", y_hat_all)
# -

# ## Plot prediction with interval

saveit = 1
plt.figure(1, figsize=(10,8))
p_names = ["Kr", "CaL", "K1", "Na", "NaL"]
p_idx = 4
for i in range(X_test.shape[0]):
    plt.plot(y_test[i, p_idx], y_mean[i, p_idx],'ko', alpha=0.6, markersize=4)
    plt.plot([y_test[i, p_idx], y_test[i, p_idx]],[y_lower[i, p_idx], y_upper[i, p_idx]], 'r-', lw=0.6, zorder=0)
# plt.xticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
# plt.yticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
plt.xlabel('$\lambda_{true}$')
plt.ylabel('$\lambda_{estimate}$')
plt.legend(["Mean estimate", "Standard deviation"])
plt.grid()
# plt.axis('equal')
if saveit:
    plt.savefig(f"plotpi_cnn1-2d-01_ne500_{p_names[p_idx]}.png",
                dpi=300, bbox_inches="tight")

# # Validation curves

# +
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
