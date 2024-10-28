# 导入库函数，包括os、tf等等
import tensorflow as tf

# # Disable all GPUs to force the model to run on CPU
# tf.config.set_visible_devices([], 'GPU')
import numpy as np
import scipy.io as sio
import os
from model import modelx

tf.config.run_functions_eagerly(True)
DEVICE = "cuda" if tf.test.is_gpu_available() else "cpu"


# Using zero-padding to support arbitrary single-lead.
def paddingecg(ecg1, idx=0):
    l_index = np.arange(ecg1.shape[0]).reshape(-1, 1)
    h_index = idx * np.ones((ecg1.shape[0], 1)).astype(np.int32)
    index = np.hstack((l_index, h_index))
    ecg_new = tf.transpose(tf.zeros_like(tf.tile(ecg1, [1, 1, 12]), dtype=tf.float32), [0, 2, 1])
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, ecg1[:, :, 0])
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new


# "Read the data, which may need to be adjusted according to actual requirements."
# "The output dimension should be (num, length)."
def Read_ECG(datapath, lead_idx=0):
    # "datapath: The path where the data is stored, in .mat format."
    # "model: The model implemented in this study."
    # "lead_idx: The position of the input lead, I to V6 correspond to 0 to 11."
    ecg1_all = []
    ecgpaths = [os.path.join(datapath, f) for f in os.listdir(datapath) if
                f.endswith('.mat') and os.path.isfile(os.path.join(datapath, f))]

    for ecgpath in ecgpaths:
        '''
        "Read the data."
        "The following code reads the CPSC2018 dataset and selects lead I. The input data should be of shape (N, 1024, 12)."
        '''

        ecg = sio.loadmat(ecgpath)['ECG'][0][0][2]
        ecg = tf.transpose(ecg)
        ecg1 = ecg[:, lead_idx:lead_idx + 1][None, :]
        ecg1_all.append(ecg1)
    return ecg1_all


# "The process of reconstructing the model, which supports arbitrary lengths, but preferably a multiple of 1024. The sampling frequency is 500Hz."
# "If it is not 500Hz, a resampling process is required."
# "The input parameters are a single-lead ECG signal, the model, the lead position, and the signal length ecglen which is 1024."
def Reconstructing_ECG(ecgall, model, ecglen=1024, lead_idx=0):
    gen_ecg12all = []
    for ecg1 in ecgall:
        padding_len = ecglen - ecg1.shape[1] % ecglen
        ecg1_new = tf.concat([ecg1, ecg1[:, -padding_len:, :]], axis=1)
        ecg1_new = tf.cast(tf.reshape(ecg1_new, shape=(-1, 1024, 1)), dtype=tf.float32)
        ecg1_new = paddingecg(ecg1_new, lead_idx)
        gen_ecg12 = model.predict(ecg1_new)
        gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg1.shape[1] + padding_len, 12))
        gen_ecg12 = gen_ecg12[:, :-padding_len, :]
        gen_ecg12all.append(gen_ecg12[0])
    return gen_ecg12all


if __name__ == '__main__':
    lead_idx = 0
    ecglen = 1024
    # Model loading
    # https://drive.google.com/drive/folders/1m57dz-FhcQCGNoZ2wxA_sUoHgrrGRHIn?usp=sharing
    # Ignore this step after you have saved the model
    # keras_model = modelx()

    # saved_model_path = "Generator"
    #
    # # Step 2: Load the SavedModel
    # saved_model = tf.saved_model.load(saved_model_path)
    #
    # # Access variables
    # saved_variables = {var.name: var for var in saved_model.variables}
    # keras_variables = {var.name: var for var in keras_model.variables}
    #
    # # Map and assign variables
    # for keras_var in keras_model.variables:
    #     # Adjust variable names if necessary
    #     var_name = keras_var.name
    #     if var_name in saved_variables:
    #         saved_var = saved_variables[var_name]
    #         keras_var.assign(saved_var)
    #     else:
    #         print(f"Variable {var_name} not found in SavedModel.")
    #
    # # Step 3: Save the Keras model in HDF5 format
    # keras_model.save('model_2.keras')
    # # After saving the model, ignore the above steps
    
    # Load the HDF5 model
    model = tf.keras.models.load_model('model_2.keras')

    # Verify model structure and summary
    model.summary()

    # Reading your data, firstly
    datapath = "Sample_data"

    ecg1 = Read_ECG(datapath, lead_idx=lead_idx)
    print(ecg1)
    # generating 12-lead ECG
    gen_ecg12 = Reconstructing_ECG(ecg1, model=model, ecglen=ecglen, lead_idx=lead_idx)
    print(gen_ecg12)
    print("Done")
