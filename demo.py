# 导入库函数，包括os、tf等等
import tensorflow as tf

# # Disable all GPUs to force the model to run on CPU
# tf.config.set_visible_devices([], 'GPU')
import numpy as np
import scipy.io as sio
import os
from scipy.signal import resample
import matplotlib.pyplot as plt
import pandas as pd
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
def read_ecg(datapath, lead_idx=0):
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
def reconstructing_ecg(ecgall, model, ecglen=1024, lead_idx=0):
    gen_ecg12all = []
    for ecg1 in ecgall:
        padding_len = ecglen - ecg1.shape[1] % ecglen
        ecg1_new = tf.concat([ecg1, ecg1[:, -padding_len:, :]], axis=1)
        ecg1_new = tf.cast(tf.reshape(ecg1_new, shape=(-1, ecglen, 1)), dtype=tf.float32)
        ecg1_new = paddingecg(ecg1_new, lead_idx)
        gen_ecg12 = model.predict(ecg1_new)  # Shape: (num_segments, 1024, 12)

        # Reshape and remove padding
        total_length = ecg1.shape[1]
        gen_ecg12 = gen_ecg12.reshape(-1, 12)[:total_length, :]

        gen_ecg12all.append(gen_ecg12)

    gen_ecg12all = np.concatenate(gen_ecg12all, axis=0)  # Shape: (TotalSamples, 12)
    return gen_ecg12all


def resampling(ecg, original_fs, desired_fs):
    duration = len(ecg) / original_fs
    num_samples = int(duration * desired_fs)
    resampled_ecg = resample(ecg, num_samples)
    return resampled_ecg


def segment_signal(ecg, segment_length=1024):
    num_segments = len(ecg) // segment_length
    ecg_truncated = ecg[:num_segments * segment_length]
    ecg_segments = ecg_truncated.reshape(num_segments, segment_length)
    return ecg_segments


def read_h10_ecg(path, original_fs=133, desired_fs=500):
    data = pd.read_csv(path, sep=';', header=0)
    print(data.head())
    ecg_signal = data['ecg [uV]'].values
    ecg_signal = resampling(ecg_signal, original_fs, desired_fs)
    # Convert uV to mV
    ecg_signal = ecg_signal / 1000
    # Segment the signal into 1024-length segments
    ecg_segments = segment_signal(ecg_signal)
    # Expand to 12 leads
    ecg_segments_12leads = np.repeat(ecg_segments[:, :, np.newaxis], 12, axis=2)
    # Select the desired lead (though in this case, all leads are the same)
    ecg_lead = ecg_segments_12leads[:, :, lead_idx:lead_idx + 1]  # Shape: (N, 1024, 1)

    # Return as a list of arrays
    ecg_data = [ecg_lead[i][np.newaxis, :] for i in range(ecg_lead.shape[0])]

    return ecg_data


def read_h10_ecg_from_txt_files(datapath):
    ecg1_all = []
    ecgpaths = [os.path.join(datapath, f) for f in os.listdir(datapath) if
                f.endswith('.txt') and os.path.isfile(os.path.join(datapath, f))]

    for ecgpath in ecgpaths:
        ecg_data = read_h10_ecg(ecgpath)
        ecg1_all.extend(ecg_data)

    return ecg1_all


def plot_reconstructed_ecg(gen_ecg12, sample_rate=500, plot_duration=None):
    """
    Visualizes the reconstructed ECG signals.

    Parameters:
        gen_ecg12 (numpy array): Output from reconstructing_ecg, containing 12-lead ECG reconstructions.
        sample_rate (int): Sampling rate of the ECG signal in Hz, default is 500 Hz.
        plot_duration (float or None): Duration in seconds to plot. If None, plot the full signal.
    """
    total_samples = gen_ecg12.shape[0]

    if plot_duration is not None:
        # Calculate the number of samples corresponding to the desired plot duration
        num_samples_to_plot = int(plot_duration * sample_rate)
        # Ensure we don't exceed the total number of samples
        if num_samples_to_plot > total_samples:
            num_samples_to_plot = total_samples
            print(f"Plot duration exceeds signal length. Plotting full signal ({total_samples / sample_rate} seconds).")
        gen_ecg12 = gen_ecg12[:num_samples_to_plot]
    else:
        num_samples_to_plot = total_samples

    # Generate a time vector for the specified duration
    ecg_duration = num_samples_to_plot / sample_rate  # duration in seconds
    time = np.linspace(0, ecg_duration, num_samples_to_plot)

    # Plot each lead
    plt.figure(figsize=(15, 12))
    for lead in range(12):
        plt.subplot(6, 2, lead + 1)
        plt.plot(time, gen_ecg12[:, lead], label=f'Lead {lead + 1}')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.legend(loc="upper right")
        plt.tight_layout()

    plt.suptitle("Reconstructed 12-Lead ECG", y=1.02, fontsize=16)
    plt.savefig("reconstructed_ecg.png")
    plt.show()




if __name__ == '__main__':
    lead_idx = 0
    ecglen = 1024
    # Model loading
    # https://drive.google.com/drive/folders/1m57dz-FhcQCGNoZ2wxA_sUoHgrrGRHIn?usp=sharing
    # Ignore this step after you have saved the model
    keras_model = modelx()

    saved_model_path = "Generator"

    # Step 2: Load the SavedModel
    saved_model = tf.saved_model.load(saved_model_path)

    # Access variables
    saved_variables = {var.name: var for var in saved_model.variables}
    keras_variables = {var.name: var for var in keras_model.variables}

    # Map and assign variables
    for keras_var in keras_model.variables:
        # Adjust variable names if necessary
        var_name = keras_var.name
        if var_name in saved_variables:
            saved_var = saved_variables[var_name]
            keras_var.assign(saved_var)
        else:
            print(f"Variable {var_name} not found in SavedModel.")

    # Step 3: Save the Keras model in HDF5 format
    keras_model.save('model_2.keras')
    # After saving the model, ignore the above steps

    # Load the HDF5 model
    model = tf.keras.models.load_model('model_2.keras')

    # Verify model structure and summary
    model.summary()

    # Reading your data, firstly
    datapath = "data/ecg-polar-h10-set4/ecg-polar-h10-set4/Polar_H10_78887921_20210712_232913_ECG.txt"

    ecg1 = read_h10_ecg(datapath)
    # generating 12-lead ECG
    gen_ecg12 = reconstructing_ecg(ecg1, model=model, ecglen=ecglen, lead_idx=lead_idx)
    # Save the reconstructed ECG
    np.save("reconstructed_ecg.npy", gen_ecg12)
    # Load the reconstructed ECG
    gen_ecg12 = np.load("reconstructed_ecg.npy", allow_pickle=True)
    plot_reconstructed_ecg(gen_ecg12, plot_duration=10)
    print("Done")
