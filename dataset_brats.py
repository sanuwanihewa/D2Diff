import torch.utils.data
import numpy as np

#This particular setting of dataset will use flair, t2 and t1 contrasts as conditional contrasts
# to synthesize the t1ce as the target contrast. Change the order of the contrasts to train model
# on different contrast synthesis. Makes sure same dataloader (with same contrast order) is used during
# the training and testing of each contrast synthesis.
def CreateDatasetSynthesis(phase, input_path):
    cond_data1 = input_path + phase + "/flair.npy"
    data_fs_s1 = LoadDataSet(cond_data1)

    cond_data2 = input_path + phase + "/t2.npy"
    data_fs_s2 = LoadDataSet(cond_data2)

    cond_data3 = input_path + phase + "/t1.npy"
    data_fs_s3 = LoadDataSet(cond_data3)

    target_data = input_path + phase + "/t1ce.npy"
    data_fs_s4 = LoadDataSet(target_data)

    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2), torch.from_numpy(data_fs_s3), torch.from_numpy(data_fs_s4))
    return dataset


def LoadDataSet(load_dir, padding=True, Norm=True):

    # Load the Numpy array
    data=np.load(load_dir)

    # Transpose and expand dimensions if necessary
    if data.ndim == 3:
        data = np.expand_dims(np.transpose(data, (0, 2, 1)), axis=1)
    else:
        data = np.transpose(data, (1, 0, 3, 2))

    data = data.astype(np.float32)

    if padding:
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

    if Norm:
        data = (data - 0.5) / 0.5

    return data