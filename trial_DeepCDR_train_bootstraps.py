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
from create_data_generator import data_generator, batch_predict

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specific imports
from model_params_def import train_params # [Req]

training = False
dropout1 = 0.10
dropout2 = 0.20
## get the model architecture
def deepcdrgcn(dict_features, dict_adj_mat, samp_drug, samp_ach, cancer_dna_methy_model, cancer_gen_expr_model, cancer_gen_mut_model, training = training, dropout1 = dropout1, dropout2 = dropout2):
    
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
    final_out_layer = tf.keras.layers.Dense(1, activation = "linear")
    final_out = final_out_layer(x)
    simplecdr = tf.keras.models.Model([input_gcn_features, input_norm_adj_mat, input_gen_expr1,
                                   input_gen_methy1, input_gen_mut1], final_out)
    
    return simplecdr


modelpath = frm.build_model_path(model_file_name="DeepCDR_model", model_file_format="", model_dir="new_result")
print(modelpath)

train_data_fname = frm.build_ml_data_file_name(data_format=".csv", stage="train")  # [Req]
val_data_fname = frm.build_ml_data_file_name(data_format=".csv", stage="val")  # [Req]

print(train_data_fname)
print(val_data_fname)

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

train_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]
valid_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

samp_drug = valid_keep["Drug_ID"].unique()[-1]
samp_ach = np.array(valid_keep["Cell_Line"].unique()[-1])

print(samp_drug)
print(samp_ach)

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

train_gcn_feats = np.array(train_gcn_feats).astype("float32")
valid_gcn_feats = np.array(valid_gcn_feats).astype("float32")

train_adj_list = np.array(train_adj_list).astype("float32")
valid_adj_list = np.array(valid_adj_list).astype("float32")

# create a data generator for the train data
batch_size = 32
generator_batch_size = 32

train_gen =  data_generator(train_gcn_feats, train_adj_list, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1), 
    train_keep["Cell_Line"].values.reshape(-1,1), train_keep["AUC"].values.reshape(-1,1), batch_size, shuffle=True, peek=True, verbose=False)

val_gen =  data_generator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), 
    valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["AUC"].values.reshape(-1,1), generator_batch_size, peek=True)

steps_per_epoch = int(np.ceil(len(train_gcn_feats) / batch_size))
train_steps = int(np.ceil(len(train_gcn_feats) / generator_batch_size))
validation_steps = int(np.ceil(len(valid_gcn_feats) / generator_batch_size))

training = False
dropout1 = 0.10
dropout2 = 0.20

# Okay, what all needs to happen inside the for loop? Model defining, compiling, fitting, prediction, save the metrics and the model. Maybe we should do folderwise model prediction saving and the model saving, that way, we probably can use the improvelib code as it is. That would be much less work, let's do that?

# Since we thought about making a directory for each bootsrap sample, we can do this with os python package - os.makedir() or os.makedirs()

# We may also need to create a mainfolder to store in the bootstrap results

os.makedirs("bootstrap_results", exist_ok = True)

check = deepcdrgcn(dict_features, dict_adj_mat, samp_drug, samp_ach, cancer_dna_methy_model, cancer_gen_expr_model, cancer_gen_mut_model,  training = training, dropout1 = dropout1, dropout2 = dropout2)

# fit the model              
epoch_num = 150
patience_val = 10

# compile the model
lr = 0.001
check.compile(loss = tf.keras.losses.MeanSquaredError(), 
                      # optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False), 
                    metrics = [tf.keras.metrics.RootMeanSquaredError()])


check.fit(train_gen, validation_data = val_gen, epochs = epoch_num, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
         callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = patience_val, restore_best_weights=True, 
                                                     mode = "min") ,validation_batch_size = generator_batch_size)

# create a data generator for the train data
batch_size = 32
generator_batch_size = 32

# get the predictions on the validation set
val_steps = int(np.ceil(len(valid_gcn_feats) / generator_batch_size))
preds_val, target_val = batch_predict(check, data_generator(valid_gcn_feats, valid_adj_list, valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["Cell_Line"].values.reshape(-1,1), valid_keep["AUC"].values.reshape(-1,1), generator_batch_size, shuffle = False), val_steps)   

print(preds_val.shape)
print(target_val.shape)

frm.store_predictions_df(
        y_true=target_val, 
        y_pred=preds_val, 
        stage="val",
        y_col_name="auc",
        output_dir='new_result',
        input_dir='exp_result'
    )

val_scores = frm.compute_performance_scores(
        y_true=target_val, 
        y_pred=preds_val, 
        stage="val",
        metric_type='regression',
        output_dir= 'new_result'
    )

# save the model?
check.save(os.path.join(modelpath, "DeepCDR_model"))


# Since this code is working, we may need to adjust this code to store all the values/as we may not be directly able to adjust the code for improve lib? Maybe we cab adjust the code there to, but currently it seems like so much work.

