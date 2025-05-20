import numpy as np
from scipy.io import loadmat
import os

import numpy as np
from sklearn.discriminant_analysis import _cov
import scipy

from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
import random
from tqdm import tqdm

def load_eeg_data_seed(folder_path, save_path, do_preprocessing=True):
    """
    Load EEG data and labels from a specified folder.

    Parameters:
    folder_path (str): The path to the folder containing the subject folders.
    save_path (str): The path to save the processed data.
    do_preprocessing (bool): Whether to perform denoising and resampling.

    Returns:
    None
    """
    # 定义EEG数据和标签文件夹路径
    eeg_folder_path = os.path.join(folder_path, 'Preprocessed_EEG')

    # 加载标签数据
    label_file_path = os.path.join(eeg_folder_path, 'label.mat')
    label_data = loadmat(label_file_path)
    label_array = np.array(label_data['label'])  # 假设标签存储在名为'label'的变量中

    # 转换标签形状
    new_label_array = np.zeros((15, 3), dtype=int)
    for i in range(15):
        if label_array[0, i] == -1:
            new_label_array[i] = [1, 0, 0]
        elif label_array[0, i] == 0:
            new_label_array[i] = [0, 1, 0]
        elif label_array[0, i] == 1:
            new_label_array[i] = [0, 0, 1]
        else:
            raise ValueError(f"Invalid label value: {label_array[0, i]}")

    # 遍历每个subject编号
    for subject_number in range(1, 16):  # subject编号从1到15
        subject_eeg_data = []  # 用于存储当前subject的所有EEG数据
        subject_labels = []  # 用于存储当前subject的所有标签

        # 遍历EEG数据文件夹，找到当前subject的所有date对应的文件
        subject_eeg_files = [f for f in os.listdir(eeg_folder_path) if f.startswith(f'{subject_number}_') and f != 'label.mat']

        for eeg_file in subject_eeg_files:
            # 加载EEG数据
            eeg_file_path = os.path.join(eeg_folder_path, eeg_file)
            eeg_data = loadmat(eeg_file_path)

            # 提取15个变量
            eeg_variables = []
            for i in range(1, 16):
                # 动态构建变量名
                variable_name = [key for key in eeg_data.keys() if key.endswith(f'_eeg{i}')][0]
                eeg_variables.append(eeg_data[variable_name])

            # 确保提取到15个变量
            if len(eeg_variables) != 15:
                raise ValueError(f"Expected 15 EEG variables in file {eeg_file}, but found {len(eeg_variables)}")


            # 找到最长的timepoint
            max_timepoint = max([var.shape[1] for var in eeg_variables])

            # 对每个变量进行填充，使其时间维度一致
            eeg_variables_padded = [np.pad(var, ((0, 0), (0, max_timepoint - var.shape[1])), mode='constant') for var in eeg_variables]

            # 将15个变量合并为一个数组，形状为(15, 62, max_timepoint)
            eeg_array = np.stack(eeg_variables_padded, axis=0)

            # Denoise EEG data
            fs = 500  # Sampling frequency
            new_fs = 100  # New sampling frequency
            lowcut = 0.5  # Low frequency cutoff
            highcut = 50  # High frequency cutoff
            order = 3  # Filter order

            if do_preprocessing:
                print(eeg_array.shape)
                eeg_array = denoise_and_resample_eeg_data(eeg_array, fs, lowcut, highcut, order, new_fs)
                print(eeg_array.shape)

            # 将当前date的EEG数据添加到列表中
            subject_eeg_data.append(eeg_array)

            # 获取当前EEG文件对应的标签
            eeg_file_labels = new_label_array
            subject_labels.append(eeg_file_labels)

        # 确保有三个date的数据
        if len(subject_eeg_data) != 3:
            raise ValueError(f"Expected 3 dates for subject {subject_number}, but found {len(subject_eeg_data)}")

        # 将三个date的EEG数据合并为一个数组，形状为(45, 62, max_timepoint)
        subject_eeg_data = np.concatenate(subject_eeg_data, axis=0)

        # 将三个date的标签合并为一个数组，形状为(45, 3)
        subject_labels = np.tile(new_label_array, (3, 1))

        # 创建保存路径
        target_subject_folder = os.path.join(save_path, f'subject{subject_number}')
        if not os.path.exists(target_subject_folder):
            os.makedirs(target_subject_folder)

        # 保存为npy文件
        eeg_save_path = os.path.join(target_subject_folder, f'subject{subject_number}_eeg.npy')
        label_save_path = os.path.join(target_subject_folder, f'subject{subject_number}_eeg_label.npy')

        subject_eeg_data=np.transpose(subject_eeg_data, (2, 1, 0))
        subject_labels=np.transpose(subject_labels, (1, 0))

        
        np.save(eeg_save_path, subject_eeg_data)
        np.save(label_save_path, subject_labels)

        print(f"Saved EEG data to {eeg_save_path}")
        print(f"Saved label data to {label_save_path}")

def load_eeg_data(folder_path,save_path,do_preprocessing=True):
    """
    Load EEG data and labels from a specified folder.

    Parameters:
    folder_path (str): The path to the folder containing the subject folders.

    Returns:
    tuple: A tuple containing two NumPy arrays: all_eeg_arrays and all_label_arrays.
    """
    # 获取文件夹列表
    subject_folders = [f for f in os.listdir(folder_path) if f.startswith('subject')]

    '''
    # 初始化列表以存储所有eeg_array和label_array
    all_eeg_arrays = []
    all_label_arrays = []
    '''

    for subject_folder in subject_folders:
        # 提取文件夹名称中的数字，并去除前导0
        subject_number = subject_folder.replace('subject', '').lstrip('0')
        
        # 定义文件路径
        eeg_file_path = os.path.join(folder_path, subject_folder, 'EEG', f'subject{subject_number}_eeg.mat')
        label_file_path = os.path.join(folder_path, subject_folder, 'EEG', f'subject{subject_number}_eeg_label.mat')

        # 加载EEG数据
        eeg_data = loadmat(eeg_file_path)
        # 加载标签数据
        label_data = loadmat(label_file_path)

        # 检查seg或seg1是否存在
        if 'seg' in eeg_data:
            eeg_array = np.array(eeg_data['seg'])  # 转换为NumPy数组
        elif 'seg1' in eeg_data:
            eeg_array = np.array(eeg_data['seg1'])  # 转换为NumPy数组
        else:
            raise ValueError(f"Neither 'seg' nor 'seg1' found in EEG data file {subject_folder}")

        # 假设标签数据存储在名为'label'的变量中
        label_array = np.array(label_data['label'])  # 转换为NumPy数组

        # 将当前的eeg_array和label_array添加到列表中
        #all_eeg_arrays.append(eeg_array)
        #all_label_arrays.append(label_array)

        # Denoise EEG data
        fs = 500  # Sampling frequency
        new_fs = 100  # New sampling frequency
        lowcut = 0.5  # Low frequency cutoff
        highcut = 50  # High frequency cutoff
        order = 3  # Filter order

        if do_preprocessing:
            print(eeg_array.shape)
            eeg_array = denoise_and_resample_eeg_data(eeg_array, fs, lowcut, highcut, order,new_fs)
            print(eeg_array.shape)

        target_subject_folder = os.path.join(save_path,  f'subject{subject_number}')
        if not os.path.exists(target_subject_folder):
            os.makedirs(target_subject_folder)

        # 保存为npy文件
        eeg_save_path = os.path.join(target_subject_folder, f'subject{subject_number}_eeg.npy')
        label_save_path = os.path.join(target_subject_folder, f'subject{subject_number}_eeg_label.npy')

        #np.save(eeg_save_path, eeg_array)
        #np.save(label_save_path, label_array)

        print(f"Saved EEG data to {eeg_save_path}")
        print(f"Saved label data to {label_save_path}")

    print("EEG data and labels have been prepared.")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, padlen=None):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, padlen=padlen)
    return y

def denoise_and_resample_eeg_data(eeg_data, fs=500, lowcut=0.5, highcut=50, order=5, new_fs=250):
    """
    Denoise EEG data using a bandpass filter and resample to a new sampling frequency.

    Parameters:
    eeg_data (np.array): The EEG data to be denoised with shape (samples, channels, trials).
    fs (int): Original sampling frequency of the EEG data.
    lowcut (float): Low frequency cutoff for the bandpass filter.
    highcut (float): High frequency cutoff for the bandpass filter.
    order (int): Order of the butterworth filter.
    new_fs (int): New sampling frequency after resampling.

    Returns:
    np.array: Denoised and resampled EEG data with the new sampling rate.
    """
    num_samples, num_channels, num_trials = eeg_data.shape
    new_num_samples = int(num_samples * new_fs / fs)
    resampled_data = np.zeros((new_num_samples, num_channels, num_trials))

    # Apply bandpass filter and resample each trial
    for trial in tqdm(range(num_trials), desc="Processing Trials"):
        for channel in range(num_channels):
            # Denoise
            denoised_channel = butter_bandpass_filter(eeg_data[:, channel, trial], lowcut, highcut, fs, order)
            # Resample
            resampled_data[:, channel, trial] = resample(denoised_channel, new_num_samples)
    
    return resampled_data

def load_data_from_file(args,folder_path,save_path, preprocessing_data_path):
    """
    从指定目录加载预处理数据或加载原始数据并进行预处理。
    
    :param args: 参数对象，包含以下属性：
        - preprocessed_max_subject: 最大主体编号，默认为 42
        - folder_path: 原始数据所在的目录路径
        - save_path: 预处理数据保存的目录路径
        - do_mvnn: 是否进行多变量归一化处理，默认为 False
    :param preprocessing_data_path: 预处理文件所在的目录路径
    :return: 加载的数据列表
    """
    data_list = []
    label_list = []
    
    # 检查目录是否存在
    if not os.path.exists(preprocessing_data_path):
        print(f"目录 {preprocessing_data_path} 不存在，将加载原始数据并进行预处理。")
        # 加载原始数据并进行预处理
        load_eeg_data_seed(folder_path, save_path, do_preprocessing=False)
        preprocessing_data_path = save_path  # 更新预处理数据路径为保存路径
    else:
        print(f"从目录 {preprocessing_data_path} 加载预处理数据。")
    
    # 遍历预处理数据
    for i in range(1, args.preprocessed_max_subject + 1):
        eeg_path = os.path.join(preprocessing_data_path,f'subject{i}' ,f"subject{i}_eeg.npy")
        label_path = os.path.join(preprocessing_data_path,f'subject{i}' ,f"subject{i}_eeg_label.npy")
        if os.path.exists(eeg_path) and os.path.exists(label_path):
            eeg_data = np.load(eeg_path)  # 假设数据以 .npy 格式存储
            label_data = np.load(label_path)
            data_list.append(eeg_data)
            label_list.append(label_data)
        else:
            print(f"文件 {preprocessing_data_path} 不存在，将加载原始数据并进行预处理。")
            # 加载原始数据并进行预处理
            load_eeg_data_seed(folder_path, save_path, do_preprocessing=False)
            # 再次检查文件是否存在
            if os.path.exists(eeg_path) and os.path.exists(label_path):
                eeg_data = np.load(eeg_path)  # 假设数据以 .npy 格式存储
                label_data = np.load(label_path)
                data_list.append(eeg_data)
                label_list.append(label_data)
            else:
                print(f"文件 {file_path} 仍然不存在，跳过该主体。")
    
    return data_list,label_list

# 定义自定义 Dataset 类
class Dataset_prepare(Dataset):
    def __init__(self, inputs, labels):
        """
        初始化自定义数据集。
        :param inputs: 输入数据的列表。
        :param labels: 标签数据的列表。
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取数据项。
        :param idx: 数据的索引。
        :return: 输入数据和对应的标签。
        """
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return input_data, label

import numpy as np

def split_dataset(data, labels, test_size=0.2, random_seed=None):
    """
    将数据集和标签切分为训练集和测试集。
    
    参数:
        data (numpy.ndarray): 完整数据集。
        labels (numpy.ndarray): 数据集对应的标签。
        test_size (float): 测试集占总数据集的比例，默认为0.2。
        random_seed (int): 随机数种子，用于确保切分结果可复现。如果不设置，则每次切分结果可能不同。
    
    返回:
        train_data (numpy.ndarray): 训练数据。
        train_labels (numpy.ndarray): 训练标签。
        test_data (numpy.ndarray): 测试数据。
        test_labels (numpy.ndarray): 测试标签。
    """
    # 检查输入数据和标签的维度是否匹配
    if data.shape[0] != labels.shape[0]:
        raise ValueError("数据集和标签的样本数量不匹配！")
    
    # 设置随机数种子
    if random_seed is not None:
        print(f"使用的随机数种子为: {random_seed}")
        np.random.seed(random_seed)
    else:
        print("未设置随机数种子，切分结果可能每次不同。")
        # 获取当前的随机种子
        current_state = np.random.get_state()
        print("当前随机种子:", current_state)
    
    # 计算测试集的样本数量
    num_samples = data.shape[0]
    test_samples = int(num_samples * test_size)
    
    # 随机打乱索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # 划分训练集和测试集
    train_indices = indices[test_samples:]
    test_indices = indices[:test_samples]
    
    # 提取训练集和测试集的数据和标签
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, train_labels, test_data, test_labels

def cut_and_extend_data(data, labels, l):
    """
    将数据集按长度 l 切割并扩充，返回切割后的数据和对应的标签。

    参数：
        data (numpy.ndarray): 原始数据，形状为 (样本数, 时间步长, 特征数)。
        labels (numpy.ndarray): 原始标签，形状为 (样本数, 类别数)。
        l (int): 每个子样本的长度。

    返回：
        extended_data (numpy.ndarray): 切割后的数据，形状为 (x, l, 特征数)。
        extended_labels (numpy.ndarray): 切割后的标签，形状为 (x, 类别数)。
    """
    extended_data = []
    extended_labels = []

    for sample, label in zip(data, labels):
        sample_length = sample.shape[0]  # 当前样本的时间步长
        num_segments = sample_length // l  # 计算可以切割的段数

        for i in range(num_segments):
            start = i * l
            end = start + l
            segment = sample[start:end, :]  # 切割出长度为 l 的子样本
            extended_data.append(segment)
            extended_labels.append(label)  # 对应的标签

    # 转换为 NumPy 数组
    extended_data = np.array(extended_data)
    extended_labels = np.array(extended_labels)

    return extended_data, extended_labels

def expand_and_shuffle(data, labels, x):
    """
    将输入数据和标签扩展x倍并打乱顺序，同时保持数据和标签的对应关系。

    参数:
        data (numpy.ndarray): 输入数据，形状为 (n_samples, n_features)
        labels (numpy.ndarray): 输入标签，形状为 (n_samples,)
        x (int): 扩展倍数

    返回:
        new_data (numpy.ndarray): 扩展并打乱后的数据
        new_labels (numpy.ndarray): 扩展并打乱后的标签
    """
    # 检查输入数据和标签的长度是否一致
    if len(data) != len(labels):
        raise ValueError("数据和标签的长度不一致！")

    # 扩展数据和标签
    new_data = np.tile(data, (x, 1, 1))  # 按行重复x次
    new_labels = np.tile(labels, (x, 1))   # 重复标签x次

    # 打乱顺序
    indices = np.arange(new_data.shape[0])
    np.random.shuffle(indices)
    new_data = new_data[indices]
    new_labels = new_labels[indices]

    return new_data, new_labels


