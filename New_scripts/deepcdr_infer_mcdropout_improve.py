# Let's get the notebook codes over here in the python script
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
import numpy as np
import pandas as pd
import os
import json
import sys
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union
# from create_data_generator import data_generator, batch_predict
from New_data_generator_with_tf import DataGenerator, batch_predict

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specific imports
from model_params_def import infer_params # [Req]

# # device ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params):
    """ Execute specified model training.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.

    :return: List of floats evaluating model predictions according to
             specified metrics_list.
    :rtype: float list
    """
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    # import the preprocessed data
    # specify the directory where preprocessed data is stored
    data_dir = params['input_data_dir']

    # load models for preprocessed data
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

    test_keep = pd.read_csv(os.path.join(data_dir, "test_y_data.csv"))
    test_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]

    test_gcn_feats = []
    test_adj_list = []
    for drug_id in test_keep["Drug_ID"].values:
        test_gcn_feats.append(dict_features[drug_id])
        test_adj_list.append(dict_adj_mat[drug_id])

    test_gcn_feats = np.array(test_gcn_feats).astype("float32")
    test_adj_list = np.array(test_adj_list).astype("float32")


    # Okay, have to think from here as to what to include
    
    # include the sample drug
    samp_drug = test_keep["Drug_ID"].unique()[-1]
    samp_ach = np.array(test_keep["Cell_Line"].unique()[-1])

    # need to load the model
    
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"], model_file_format=params["model_file_format"],
                                     model_dir=params["input_model_dir"]) # [Req]
    model_path = os.path.join(modelpath, "DeepCDR_model")
    print(model_path)
    CCLE_model = tf.keras.models.load_model(model_path)

    # redefine the model
    training = True
    dropout1 = 0.10
    dropout2 = 0.20

    input_gcn_features = tf.keras.layers.Input(shape = (dict_features[samp_drug].shape[0], 75))
    input_norm_adj_mat = tf.keras.layers.Input(shape = (dict_adj_mat[samp_drug].shape[0], dict_adj_mat[samp_drug].shape[0]))
    mult_1 = tf.keras.layers.Dot(1)([input_norm_adj_mat, input_gcn_features])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_1)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out)
    mult_2 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_2)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out)

    dense_layer_gcn = tf.keras.layers.Dense(100, activation = "relu")
    mult_3 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_out = dense_layer_gcn(mult_3)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out)

    dense_out = tf.keras.layers.GlobalAvgPool1D()(dense_out)
    # All above code is for GCN for drugs

    # methylation data
    input_gen_methy1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_methy = cancer_dna_methy_model(input_gen_methy1)
    input_gen_methy.trainable = False
    gen_methy_layer = tf.keras.layers.Dense(256, activation = "tanh")
    
    gen_methy_emb = gen_methy_layer(input_gen_methy)
    gen_methy_emb = tf.keras.layers.BatchNormalization()(gen_methy_emb)
    gen_methy_emb = tf.keras.layers.Dropout(dropout1)(gen_methy_emb)
    gen_methy_layer = tf.keras.layers.Dense(100, activation = "relu")
    gen_methy_emb = gen_methy_layer(gen_methy_emb)

    # gene expression data
    input_gen_expr1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_expr = cancer_gen_expr_model(input_gen_expr1)
    input_gen_expr.trainable = False
    gen_expr_layer = tf.keras.layers.Dense(256, activation = "tanh")
    
    gen_expr_emb = gen_expr_layer(input_gen_expr)
    gen_expr_emb = tf.keras.layers.BatchNormalization()(gen_expr_emb)
    gen_expr_emb = tf.keras.layers.Dropout(dropout1)(gen_expr_emb)
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
    simplecdr = tf.keras.models.Model([input_gcn_features, input_norm_adj_mat, input_gen_expr1, input_gen_methy1, input_gen_mut1], final_out)

    weights_train = CCLE_model.get_weights()
    simplecdr.set_weights(weights_train)
    weights_new = simplecdr.get_weights()
    print(len(weights_train), len(weights_new))
    
    generator_batch_size = 32
    all_predicted_values = []
    for i in range(25):
        preds_test, target_test = batch_predict(simplecdr, DataGenerator(test_gcn_feats, test_adj_list, test_keep["Cell_Line"].values.reshape(-1,1),
                                                                         test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1),
                                                                         test_keep["AUC"].values.reshape(-1,1), generator_batch_size, shuffle = False))
        all_predicted_values.append(preds_test)

    preds_df = pd.DataFrame(all_predicted_values)
    preds_df_final = preds_df.T
    print(preds_df_final.head())
    preds_df_final.columns = ['predicted_vals_' + str(i + 1) for i in range(preds_df_final.shape[1])]
    combined_df = pd.concat((preds_df_final, pd.DataFrame(target_test, columns = ['target_auc'])), axis = 1)
    print(combined_df.head())

    li_test = np.percentile(preds_df_final, axis = 1, q = (2.5, 97.5))[0,:].reshape(-1,1)     
    ui_test = np.percentile(preds_df_final, axis = 1, q = (2.5, 97.5))[1,:].reshape(-1,1)   

    width_test = ui_test - li_test
    avg_width_test = width_test.mean(0)[0]
    print(avg_width_test)

    ind_test = (target_test >= li_test) & (target_test <= ui_test)
    coverage_test= ind_test.mean(0)[0]
    print(coverage_test)

    # store the width and coverage in a json file
    # Store the outputs in a dictionary
    mc_dropout_data = {"Width of CIs - mc dropout": avg_width_test, "Coverage of CIs - mc dropout": coverage_test}
    
    # Specify the full file path
    json_file_path = os.path.join(params['output_dir'], "CI_information_mc_dropout.json")

    # Write the dictionary to a JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(mc_dropout_data, json_file, indent=4)

    # we need to also store the metrics for the averages to be improve compliant, let's work on those
    # compute the scores
    all_preds_array = np.array(preds_df_final)
    test_averaged_preds_array = np.mean(all_preds_array, axis = 1)
    print(test_averaged_preds_array.shape, target_test.shape)

    frm.store_predictions_df(
        y_true = target_test, 
        y_pred = test_averaged_preds_array, 
        stage = "test",
        y_col_name = params["y_col_name"],
        output_dir = params['output_dir'],
        input_dir = params['input_data_dir'])

    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
        y_true = target_test, 
        y_pred = test_averaged_preds_array, 
        stage = "test",
        metric_type = params["metric_type"],
        output_dir = params['output_dir']) 

# [Req]
def main(args):
    # [Req]
    additional_definitions = infer_params
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir = filepath,
        default_config = "deepcdr_params.txt",
        additional_definitions = additional_definitions
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])










