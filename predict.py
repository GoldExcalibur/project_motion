import torch
import argparse
import os
import numpy as np
import random
from transform import generate_penguin, mean_distance
from common import config
from dataset import MotionDataset
from model import CycleGANModel
from visualization import frames2video
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--h_path', type=str, default=None, required=False)
    parser.add_argument('-p2', '--nh_path', type=str, default=None, required=False)
    parser.add_argument('-e', '--epoch', type=str, default=99, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('-o', '--save_dir', type=str, default="./results", required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config.isTrain = False

    # prepare data
    test_dataset = MotionDataset("test")

    if args.h_path is None:
        h_path = test_dataset.all_paths[random.randint(0, len(test_dataset) - 1)]
    else:
        h_path = args.h_path

    if args.nh_path is None:
        nh_path = test_dataset.all_paths[random.randint(0, len(test_dataset) - 1)]
    else:
        nh_path = args.nh_path

    h_motion3d = np.load(h_path)
    nh_motion3d_gt = np.load(nh_path)
    nh_motion3d = generate_penguin(nh_motion3d_gt)
    h_motion3d_gt = generate_penguin(h_motion3d)

    gt = {'fake_A': nh_motion3d_gt, 'fake_B': h_motion3d_gt}

    data = {"A": test_dataset.preprocess(h_motion3d, "h").unsqueeze(0),
            "B": test_dataset.preprocess(nh_motion3d, "nh").unsqueeze(0)}

    # create network
    net = CycleGANModel(config)
    net.load_networks(args.epoch)

    net.eval()

    # forwarding
    with torch.no_grad():
        net.set_input(data)
        net.forward()
        ret = net.infer()

    # post precessing
    results = {}
    for k, v in ret.items():
        phase = 'h' if k[-1] == 'A' else 'nh'
        motion3d = test_dataset.preprocess_inv(v.detach().cpu().numpy()[0], phase)
        results[k] = motion3d

    dis_A2B = mean_distance(results['fake_A'], gt['fake_A'], 0)
    dis_B2A = mean_distance(results['fake_B'], gt['fake_B'], 0)
    print(dis_A2B, dis_B2A)

    # save results

    if args.save_dir is not None:
        ensure_dir(args.save_dir)

        def get_name(path):
            return ('-'.join(path.split('/')[-3:])).split('.')[0]

        save_name = 'h-' + get_name(h_path) + '_' + 'nh-' + get_name(nh_path)
        save_sub = os.path.join(args.save_dir, save_name)
        ensure_dir(save_sub)

        for k, v in results.items():
            save_path = os.path.join(save_sub, k + '.mp4')
            frames2video(v, save_path)
            print("{} saved at {}".format(k, save_path))

        for k, v in gt.items():
            save_path = os.path.join(save_sub, k + '_gt.mp4')
            frames2video(v, save_path)
            print('{} saved at {}'.format(k, save_path))


if __name__ == "__main__":
    main()
