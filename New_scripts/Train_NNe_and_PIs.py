import os
import json
import sys
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
# generator related imports
from New_data_generator_with_tf import DataGenerator, batch_predict
from tensorflow.keras.utils import Sequence

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specific imports
from model_params_def import train_params # [Req]

# specify the directory where preprocessed data is stored
data_dir = 'exp_result'

cancer_gen_expr_model = tf.keras.models.load_model(os.path.join(data_dir,"cancer_gen_expr_model"))
cancer_gen_mut_model = tf.keras.models.load_model(os.path.join(data_dir, "cancer_gen_mut_model"))
cancer_dna_methy_model = tf.keras.models.load_model(os.path.join(data_dir, "cancer_dna_methy_model"))

cancer_gen_expr_model.trainable = False
cancer_gen_mut_model.trainable = False
cancer_dna_methy_model.trainable = False

with open(os.path.join(data_dir, "drug_features.pickle"),"rb") as f:
        dict_features = pickle.load(f)

with open(os.path.join(data_dir, "norm_adj_mat.pickle"),"rb") as f:
        dict_adj_mat = pickle.load(f)

train_keep = pd.read_csv(os.path.join(data_dir, "train_y_data.csv"))
valid_keep = pd.read_csv(os.path.join(data_dir, "val_y_data.csv"))
test_keep = pd.read_csv(os.path.join(data_dir, "test_y_data.csv"))
print(train_keep.shape, valid_keep.shape, test_keep.shape)

train_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
valid_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
test_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

samp_drug = valid_keep["Drug_ID"].unique()[-1]
samp_ach = np.array(valid_keep["Cell_Line"].unique()[-1])

train_gcn_feats = []
train_adj_list = []
for drug_id in train_keep["Drug_ID"].values:
    train_gcn_feats.append(dict_features[drug_id])
    train_adj_list.append(dict_adj_mat[drug_id])

valid_gcn_feats = []
valid_adj_list = []
for drug_id in valid_keep["Drug_ID"].values:
    valid_gcn_feats.append(dict_features[drug_id])
    valid_adj_list.append(dict_adj_mat[drug_id])

test_gcn_feats = []
test_adj_list = []
for drug_id in test_keep["Drug_ID"].values:
    test_gcn_feats.append(dict_features[drug_id])
    test_adj_list.append(dict_adj_mat[drug_id])

# reduce the values to float32
train_gcn_feats = np.array(train_gcn_feats).astype("float32")
valid_gcn_feats = np.array(valid_gcn_feats).astype("float32")
test_gcn_feats = np.array(test_gcn_feats).astype("float32")

train_adj_list = np.array(train_adj_list).astype("float32")
valid_adj_list = np.array(valid_adj_list).astype("float32")
test_adj_list = np.array(test_adj_list).astype("float32")

train_data_gen = DataGenerator(train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), train_keep["AUC"].values.reshape(-1,1), batch_size=32,  shuffle = False)

val_data_gen = DataGenerator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["AUC"].values.reshape(-1,1), batch_size=32,  shuffle = False)

test_data_gen = DataGenerator(test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["AUC"].values.reshape(-1,1), batch_size=32,  shuffle = False)

# location of the models
folder_path = 'bootstrap_results_all'

# name of the trained model
model_nm = 'DeepCDR_model'

all_train_predictions = []
all_valid_predictions = []
all_test_predictions = []
train_true = []
valid_true = []
test_true = []
# number of boostraps
B = 10

# start the for loop
for i in range(1, B + 1):
    # create the folder
    folder_nm = 'bootstrap_' + str(i)
    # joined path
    folder_loc = os.path.join(folder_path, folder_nm, model_nm)
    # load the model?
    model = tf.keras.models.load_model(folder_loc)
    # get the predictions on the train data
    y_train_preds, y_train_true = batch_predict(model, train_data_gen)
    y_val_preds, y_val_true = batch_predict(model, val_data_gen)
    y_test_preds, y_test_true = batch_predict(model, test_data_gen)
    all_train_predictions.append(y_train_preds)
    all_valid_predictions.append(y_val_preds)
    all_test_predictions.append(y_test_preds)
    train_true.append(y_train_true)
    valid_true.append(y_val_true)
    test_true.append(y_test_true)

# train data
all_train_preds_array = np.array(all_train_predictions)
all_train_trues = np.array(train_true)
all_train_true_mean = np.mean(all_train_trues, axis = 0)
# these do look the same, should we do an np.mean?
print(np.mean(np.round(all_train_true_mean, 8) == np.round(np.squeeze(train_keep["AUC"].values.reshape(-1,1)), 8)))

train_bts_mean = np.mean(all_train_preds_array, axis = 0)

# validation data
all_valid_preds_array = np.array(all_valid_predictions)
all_valid_trues = np.array(valid_true)
all_valid_true_mean = np.mean(all_valid_trues, axis = 0)
# these do look the same, should we do an np.mean? - Do a sanity check
print(np.mean(np.round(all_valid_true_mean, 8) == np.round(np.squeeze(valid_keep["AUC"].values.reshape(-1,1)), 8)))

valid_bts_mean = np.mean(all_valid_preds_array, axis = 0)

# test data
all_test_preds_array = np.array(all_test_predictions)
all_test_trues = np.array(test_true)
all_test_true_mean = np.mean(all_test_trues, axis = 0)
# these do look the same, should we do an np.mean?
print(np.mean(np.round(all_test_true_mean, 8) == np.round(np.squeeze(test_keep["AUC"].values.reshape(-1,1)), 8)))

test_bts_mean = np.mean(all_test_preds_array, axis = 0)

# we also need the bootstrap variance

# let's use the same function as earlier - we cannot use this as is, as what we have now is a 2D array, and not a 3D one
def equation_6_model_variance(all_preds):
    all_vars = []
    for i in range(all_preds.shape[1]):
        var = (1/(all_preds.shape[0]  - 1))*np.sum(np.square(all_preds[:,i] - np.mean(all_preds[:,i])))
        all_vars.append(var)

    return np.array(all_vars, dtype= np.float32)

# for train data
train_bts_variance = equation_6_model_variance(all_train_preds_array)
# how to alternatively compute the variance in one line
alt_train_bts_variance = np.var(all_train_preds_array, axis = 0, ddof = 1)
print(np.mean(np.round(train_bts_variance, 6) == np.round(alt_train_bts_variance, 6)))

# for validation data
valid_bts_variance = equation_6_model_variance(all_valid_preds_array)
# how to alternatively compute the variance in one line
alt_valid_bts_variance = np.var(all_valid_preds_array, axis = 0, ddof = 1)
# sanity check ot see if the variances are inded correct
print(np.mean(np.round(valid_bts_variance, 6) == np.round(alt_valid_bts_variance, 6)))

# for test data
test_bts_variance = equation_6_model_variance(all_test_preds_array)
# how to alternatively compute the variance in one line
alt_test_bts_variance = np.var(all_test_preds_array, axis = 0, ddof = 1)
# sanity check ot see if the variances are inded correct
print(np.mean(np.round(test_bts_variance, 6) == np.round(alt_test_bts_variance, 6)))


# sanity check for the means
catch_train_mean = []
for i in range(all_train_preds_array.shape[1]):
    computed_mean = np.mean(all_train_preds_array[:,i])
    catch_train_mean.append(computed_mean)

sanity_check_train_means = np.array(catch_train_mean)
print(np.mean(np.round(train_bts_mean,2) == np.round(sanity_check_train_means, 2)))

# sanity check for the validation means
catch_valid_mean = []
for i in range(all_valid_preds_array.shape[1]):
    computed_mean = np.mean(all_valid_preds_array[:,i])
    catch_valid_mean.append(computed_mean)

sanity_check_valid_means = np.array(catch_valid_mean)
print(np.mean(np.round(valid_bts_mean, 3) == np.round(sanity_check_valid_means, 3)))

# sanity check for the test means
catch_test_mean = []
for i in range(all_test_preds_array.shape[1]):
    computed_mean = np.mean(all_test_preds_array[:,i])
    catch_test_mean.append(computed_mean)

sanity_check_test_means = np.array(catch_test_mean)
print(np.mean(np.round(test_bts_mean, 3) == np.round(sanity_check_test_means, 3)))


# Compute r^2(x_i) for each bootstrap model
def compute_r_squared(y_true, y_pred, model_variance):
    residuals = (y_true - y_pred) ** 2 - model_variance
    return np.maximum(residuals, 0)

# response variable for train data
r_2_true_train = compute_r_squared(train_keep["AUC"].values, train_bts_mean, train_bts_variance)
# Count the number of zeros
num_zeros_train = np.count_nonzero(r_2_true_train == 0)
print(num_zeros_train)

# response variable for validation data
r_2_true_valid = compute_r_squared(valid_keep["AUC"].values, valid_bts_mean, valid_bts_variance)
# Count the number of zeros
num_zeros_valid = np.count_nonzero(r_2_true_valid == 0)
print(num_zeros_valid)

# response variable for test data
r_2_true_test = compute_r_squared(test_keep["AUC"].values, test_bts_mean, test_bts_variance)
# Count the number of zeros
num_zeros_test = np.count_nonzero(r_2_true_test == 0)
print(num_zeros_test)

# Training the NNE
training = False
dropout1 = 0.10
dropout2 = 0.20
## get the model architecture
def deepcdrgcn_NNe(dict_features, dict_adj_mat, samp_drug, samp_ach, cancer_dna_methy_model, cancer_gen_expr_model, cancer_gen_mut_model, training = training, dropout1 = dropout1, dropout2 = dropout2):
    
    input_gcn_features = tf.keras.layers.Input(shape = (dict_features[samp_drug].shape[0], 75))
    input_norm_adj_mat = tf.keras.layers.Input(shape = (dict_adj_mat[samp_drug].shape[0], dict_adj_mat[samp_drug].shape[0]))
    mult_1 = tf.keras.layers.Dot(1)([input_norm_adj_mat, input_gcn_features])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_1)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)
    mult_2 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_2)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)

    dense_layer_gcn = tf.keras.layers.Dense(100, activation = "relu")
    mult_3 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_out = dense_layer_gcn(mult_3)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)

    dense_out = tf.keras.layers.GlobalAvgPool1D()(dense_out)
    # All above code is for GCN for drugs

    # methylation data
    input_gen_methy1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_methy = cancer_dna_methy_model(input_gen_methy1)
    input_gen_methy.trainable = False
    gen_methy_layer = tf.keras.layers.Dense(256, activation = "tanh")
    
    gen_methy_emb = gen_methy_layer(input_gen_methy)
    gen_methy_emb = tf.keras.layers.BatchNormalization()(gen_methy_emb)
    gen_methy_emb = tf.keras.layers.Dropout(dropout1)(gen_methy_emb, training = training)
    gen_methy_layer = tf.keras.layers.Dense(100, activation = "relu")
    gen_methy_emb = gen_methy_layer(gen_methy_emb)

    # gene expression data
    input_gen_expr1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_expr = cancer_gen_expr_model(input_gen_expr1)
    input_gen_expr.trainable = False
    gen_expr_layer = tf.keras.layers.Dense(256, activation = "tanh")
    
    gen_expr_emb = gen_expr_layer(input_gen_expr)
    gen_expr_emb = tf.keras.layers.BatchNormalization()(gen_expr_emb)
    gen_expr_emb = tf.keras.layers.Dropout(dropout1)(gen_expr_emb, training = training)
    gen_expr_layer = tf.keras.layers.Dense(100, activation = "relu")
    gen_expr_emb = gen_expr_layer(gen_expr_emb)
    
    
    input_gen_mut1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_mut = cancer_gen_mut_model(input_gen_mut1)
    input_gen_mut.trainable = False
    
    reshape_gen_mut = tf.keras.layers.Reshape((1, cancer_gen_mut_model(samp_ach).numpy().shape[0], 1))
    reshape_gen_mut = reshape_gen_mut(input_gen_mut)
    gen_mut_layer = tf.keras.layers.Conv2D(50, (1, 700), strides=5, activation = "tanh")
    gen_mut_emb = gen_mut_layer(reshape_gen_mut)
    pool_layer = tf.keras.layers.MaxPooling2D((1,5))
    pool_out = pool_layer(gen_mut_emb)
    gen_mut_layer = tf.keras.layers.Conv2D(30, (1, 5), strides=2, activation = "relu")
    gen_mut_emb = gen_mut_layer(pool_out)
    pool_layer = tf.keras.layers.MaxPooling2D((1,10))
    pool_out = pool_layer(gen_mut_emb)
    flatten_layer = tf.keras.layers.Flatten()
    flatten_out = flatten_layer(pool_out)
    x_mut = tf.keras.layers.Dense(100,activation = 'relu')(flatten_out)
    x_mut = tf.keras.layers.Dropout(dropout1)(x_mut)
    
    all_omics = tf.keras.layers.Concatenate()([dense_out, gen_methy_emb, gen_expr_emb, x_mut])
    x = tf.keras.layers.Dense(300,activation = 'tanh')(all_omics)
    x = tf.keras.layers.Dropout(dropout1)(x, training = training)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x,axis=1))(x)
    x = tf.keras.layers.Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,2))(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,3))(x)
    x = tf.keras.layers.Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,3))(x)
    x = tf.keras.layers.Dropout(dropout1)(x, training = training)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout2)(x, training = training)
    final_out_layer = tf.keras.layers.Dense(1, activation = tf.keras.activations.exponential)
    final_out = final_out_layer(x)
    simplecdr = tf.keras.models.Model([input_gcn_features, input_norm_adj_mat, input_gen_expr1,
                                   input_gen_methy1, input_gen_mut1], final_out)
    
    return simplecdr

training = False
dropout1 = 0.10
dropout2 = 0.20
NNE_model = deepcdrgcn_NNe(dict_features, dict_adj_mat, samp_drug, samp_ach, cancer_dna_methy_model, cancer_gen_expr_model, cancer_gen_mut_model,  training = training, dropout1 = dropout1, dropout2 = dropout2)
# let's define the train and validation data generators, now with our output for the NNe and not the AUC value we have had earleir
train_gen_NNe = DataGenerator(train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), r_2_true_train, batch_size=32)
val_gen_NNe = DataGenerator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), r_2_true_valid, batch_size=32,  shuffle = False)
# should we be using the test data generator for predictions? - well I believe we should as some of the test datasets are very large
test_gen_NNe = DataGenerator(test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1), r_2_true_test, batch_size=32,  shuffle = False)
# compile the model
lr = 0.01
# NNE_model.compile(loss = tf.keras.losses.MeanSquaredError(), 
#                       # optimizer = tf.keras.optimizers.Adam(lr=1e-3),
#                     optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False), 
#                     metrics = [tf.keras.metrics.RootMeanSquaredError()])

# use the new custom loss?
# Define the custom loss function as described in equation 12
def correct_custom_loss(r_true, r_pred):
    # first term in equation 12
    term_1 = tf.math.log(r_pred + 1)
    # define the second term
    term_2 = r_true/r_pred
    # cost function
    cost = 0.5 * tf.reduce_mean(term_1 + term_2)

    return cost

# compile the model, and use the new custom loss function
NNE_model.compile(loss = lambda y_true, y_pred: correct_custom_loss(
                      y_true, y_pred), 
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr), 
                    metrics = [tf.keras.metrics.RootMeanSquaredError()])

# fit the model   
batch_size = 32
generator_batch_size = 32
epoch_num = 150
patience_val = 20
NNE_model.fit(train_gen_NNe, validation_data = val_gen_NNe, epochs = epoch_num, 
              callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = patience_val, restore_best_weights=True, 
                                                           mode = "min") ,validation_batch_size = generator_batch_size)

# Write the code for predictions for the test data?
error_var_test = NNE_model.predict(test_gen_NNe)
print(np.mean(error_var_test))
print(error_var_test.shape)

y_test_pred_error, y_test_true_error = batch_predict(NNE_model, test_gen_NNe)
print(np.mean(y_test_pred_error))
print("Error variances: ")
print("Preds NNe test shape (error variance): ", y_test_pred_error.shape)
print("True error variance: ", y_test_true_error.shape)

print("Model variances: ")
print("Model variances (from bootstraps): ", test_bts_variance.shape)

print("Test bts means: ")
print("Test bts means shape: ", test_bts_mean.shape)


############### Prediction intervals ###################
# lower bound
lower_l = test_bts_mean - 1.96*np.sqrt(y_test_pred_error + test_bts_variance)
# upper bound
upper_l = test_bts_mean + 1.96*np.sqrt(y_test_pred_error + test_bts_variance)

print("Any missing values for lower bound? ", np.isnan(lower_l).sum())
print("Any missing values for upper bound? ", np.isnan(upper_l).sum())

print("Shape upper PI: ", upper_l.shape)
print("Shape lower PI: ", lower_l.shape)

# width of the confidence limits
print("Prediction Interval widths: ", np.mean(upper_l - lower_l))

y_test = np.squeeze(test_keep["AUC"].values.reshape(-1,1))

# Coverage of the confidence limits
catch_true_values = []
for i in range(upper_l.shape[0]):
    true_value =  y_test[i]
    # print(true_value)
    # print(upper_l[i])
    # print(lower_l[i])
    if lower_l[i] <= true_value <= upper_l[i]:
        catch_true_values.append(True)
    else:
        catch_true_values.append(False)

print("Prediction Interval covarage: ", np.mean(catch_true_values))

# save these errors? Or do we go ahead and straight away compute the coverages and widths?
# We need to print the shapes of all bootstrap means andvariances before computing the PIs as if the shapes don't match the output will be weird.

# We need to save this model
NNE_model.save('NNe_model')