import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from transform import generate_penguin, trans_motion3d, trans_motion3d_inv
from visualization import draw_skel

random.seed(1234)
DATA_ROOT = "/data1/wurundi/cmu-mocap"
ALL_PATHS = glob.glob(os.path.join(DATA_ROOT, '*/*/*.npy'))
random.shuffle(ALL_PATHS)
TRAIN_PATHS = ALL_PATHS[:-1000]
TEST_PATHS = ALL_PATHS[-1000:]

BASE_ID = {"h": 0, "nh": 0}


class MotionDataset(Dataset):
    def __init__(self, name):
        self.data_root = DATA_ROOT

        self.name = name

        if name == "train":
            self.all_paths = TRAIN_PATHS
        else:
            self.all_paths = TEST_PATHS
            np.random.seed(322)

        self.size = len(self.all_paths) // 2

        self.paths = {"h": self.all_paths[:self.size], "nh": self.all_paths[-self.size:]}
        self.base_id = BASE_ID

        h_mean, h_std, nh_mean, nh_std = get_meanpose()
        self.mean_pose = {"h": h_mean, "nh": nh_mean}
        self.std_pose = {"h": h_std, "nh": nh_std}

    def preprocess(self, motion3d, phase):
        motion3d = trans_motion3d(motion3d, self.base_id[phase])
        motion3d = (motion3d - self.mean_pose[phase][:, :, np.newaxis]) / self.std_pose[phase][:, :, np.newaxis]
        motion3d = motion3d.reshape(-1, motion3d.shape[-1])

        motion3d = torch.Tensor(motion3d)

        return motion3d

    def preprocess_inv(self, motion3d, phase):
        motion3d = motion3d.reshape(-1, 3, motion3d.shape[-1])
        motion3d = motion3d * self.std_pose[phase][:, :, np.newaxis] + self.mean_pose[phase][:, :, np.newaxis]
        motion3d = trans_motion3d_inv(motion3d, self.base_id[phase])
        return motion3d

    def __getitem__(self, index):
        idx1, idx2 = np.random.choice(self.size, size=2)
        h_path = self.paths["h"][idx1]
        nh_path = self.paths["nh"][idx2]

        h_motion3d = np.load(h_path)
        nh_motion3d = generate_penguin(np.load(nh_path))

        input_A = self.preprocess(h_motion3d, "h")
        input_B = self.preprocess(nh_motion3d, "nh")

        return {"A": input_A, "B": input_B,
                "A_paths": h_path, "B_paths": nh_path}

    def __len__(self):
        return self.size


def get_dataloader(name, batch_size, shuffle=True, num_workers=16):
    """
    :param name: 'train' or 'validation' or 'test'
    :param batch_size: the size of an batch
    :param shuffle: whether random shuffle the data or not
    :param num_workers: the number of workers
    :return: the dataloader
    """
    dataset = MotionDataset(name)
    if name == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers)

    return dataloader


def get_meanpose():
    h_mean_path = os.path.join(DATA_ROOT, 'mean_human.npy')
    nh_mean_path = os.path.join(DATA_ROOT, 'mean_penguin.npy')
    h_std_path = os.path.join(DATA_ROOT, 'std_human.npy')
    nh_std_path = os.path.join(DATA_ROOT, 'std_penguin.npy')

    if os.path.exists(h_mean_path) and os.path.exists(nh_mean_path):
        h_mean = np.load(h_mean_path)
        h_std = np.load(h_std_path)
        nh_mean = np.load(nh_mean_path)
        nh_std = np.load(nh_std_path)
        return h_mean, h_std, nh_mean, nh_std

    size = len(TRAIN_PATHS) // 2
    h_mean, h_std = gen_meanpose(TRAIN_PATHS[:size], BASE_ID["h"])
    np.save(h_mean_path, h_mean)
    np.save(h_std_path, h_std)

    nh_mean, nh_std = gen_meanpose(TRAIN_PATHS[:size], BASE_ID["nh"], is_nh=True)
    np.save(nh_mean_path, nh_mean)
    np.save(nh_std_path, nh_std)
    return h_mean, h_std, nh_mean, nh_std


def gen_meanpose(paths, base_id, is_nh=False):
    all_motions = []
    for path in paths:
        motion3d = np.load(path)
        if is_nh:
            motion3d = generate_penguin(motion3d)
        motion3d = trans_motion3d(motion3d, base_id)
        all_motions.append(motion3d)
    all_motions = np.concatenate(all_motions, axis=2)
    mean = np.mean(all_motions, axis=2)
    std = np.std(all_motions, axis=2)
    std[np.where(std == 0)] = 1e-9
    return mean, std


if __name__ == '__main__':
    # dataloader = MotionDataset("test")
    dataloader = get_dataloader(name='test', batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data['A_paths'])
        print(data['B_paths'])
        print("-"*20)
        if i >= 3:
            break
    exit()
    print(TRAIN_PATHS[:5])
    h_mean, h_std, nh_mean, nh_std = get_meanpose()
    dataset = MotionDataset("train")

    h_meanpose = dataset.preprocess_inv(h_mean[:, :, np.newaxis], phase="h")
    nh_meanpose = dataset.preprocess_inv(nh_mean[:, :, np.newaxis], phase="nh")
    draw_skel(h_meanpose[..., 0], save_path='test_mean_h.png', phase="h")
    draw_skel(nh_meanpose[..., 0], save_path='test_mean_nh.png', phase="nh")

    # # h_mean, h_std = gen_meanpose(dataset.paths["h"], dataset.base_id["h"])
    # # print("mean", h_mean.shape)
    # # print("std", h_std.shape)
    # #
    # # meanpose = trans_motion3d_inv(h_mean[:, :, np.newaxis])
    # # print(meanpose[..., 0])
    # # draw_skel(meanpose[..., 0], save_path='test_mean.png')
    #
    # nh_mean, nh_std = gen_meanpose(dataset.paths["nh"][::100], dataset.base_id["nh"], is_nh=True)
    # print("mean", nh_mean.shape)
    # print("std", nh_std.shape)
    #
    # meanpose = trans_motion3d_inv(nh_mean[:, :, np.newaxis])
    # print(meanpose[..., 0])
    # draw_skel(meanpose[..., 0], save_path='test_mean_nh.png', phase="nh")

    # motion_nh = np.load("/data1/wurundi/cmu-mocap/074/74_14/0.npy")
    draw_skel(np.load("/data1/wurundi/cmu-mocap/074/74_14/0.npy")[:, :, 60], save_path='test_real_A_ori.png', phase='h')
    draw_skel(generate_penguin(np.load("/data1/wurundi/cmu-mocap/074/74_14/0.npy"))[:, :, 60], save_path='test_real_B_ori.png', phase='nh')
