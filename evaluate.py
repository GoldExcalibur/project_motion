

import torch
import numpy as np
import argparse
import os
import sys

from model import CycleGANModel
from dataset import get_dataloader
from utils import AverageMeter, Table, ensure_dir
from transform import mean_distance, generate_penguin
from visualization import frames2video
from collections import namedtuple, OrderedDict
import common

def get_test_config(train_config, epoch, save_dir=None):
    test_attr = ['input_nc', 'output_nc', 'ngf', 'ndf', 'n_layers_G', 'n_layers_D',
                'G_en_ks','G_de_ks', 'D_ks', 'direction', 'model_dir', 'log_dir', 'exp_name', 'device', 'stat_path', 'nr_epochs',
                'epoch', 'isTrain', 'save_dir']
    TestConfig = namedtuple('TestConfig', test_attr)

    vals = [getattr(train_config, name) for name in test_attr[:-3]]
    tcfg = TestConfig(*(vals + [epoch, False, save_dir]))
    return tcfg


def get_gt(h_path, nh_path):
    h_motion3d = np.load(h_path)
    nh_motion3d_gt = np.load(nh_path)
    nh_motion3d = generate_penguin(nh_motion3d_gt)
    h_motion3d_gt = generate_penguin(h_motion3d)
    gt = {'fake_A': nh_motion3d_gt, 'fake_B': h_motion3d_gt}
    return gt

def evaluate(test_config):
    def get_name(path):
        print(path)
        return ('-'.join(path.split('/')[-3:])).split('.')[0]

    net = CycleGANModel(test_config)
    net.load_networks(test_config.epoch)

    net.eval()

    test_data = get_dataloader('test', batch_size = 1)
    dataset_size = len(test_data)
    print('Test Data size:', dataset_size)

    phases = ['h', 'nh']
    phases = phases if test_config.direction == 'AtoB' else phases[::-1]
    phases_dict = dict(zip(['A', 'B'], phases))
    save_dir  = test_config.save_dir

    distance_dict = {'A2B': [], 'B2A': []}
    with torch.no_grad():
        for i, data in enumerate(test_data):
            net.set_input(data)
            net.forward()
            ret = net.infer()
            preds = {}

            h_path, nh_path = data['A_paths'][0], data['B_paths'][0]
            gt = get_gt(h_path, nh_path)
            save_name = 'h-' + get_name(h_path) + '_' + 'nh-' + get_name(nh_path)
            if save_dir is not None:
                save_sub = os.path.join(save_dir, save_name)
                ensure_dir(save_sub)

            for k, v in ret.items():
                v_npy = test_data.dataset.preprocess_inv(v.detach().cpu().numpy()[0], phases_dict[k[-1]])
                preds[k] = v_npy

                if save_dir is not None:
                    frames2video(v_npy, os.path.join(save_sub, k + '.mp4'))

            dis_A = mean_distance(gt['fake_A'], preds['fake_A'], 0)
            dis_B = mean_distance(gt['fake_B'], preds['fake_B'], 0)
            distance_dict['A2B'].append(dis_A)
            distance_dict['B2A'].append(dis_B)
            print(save_name, dis_A, dis_B)

    A2B_dis = np.array(distance_dict['A2B'])
    B2A_dis = np.array(distance_dict['B2A'])
    A2B_mean, A2B_std = np.mean(A2B_dis), np.std(A2B_dis)
    B2A_mean, B2A_std = np.mean(B2A_dis), np.std(B2A_dis)
    print('A2B mean {:.3f} std {:.3f}'.format(A2B_mean, A2B_std))
    print('B2A mean {:.3f} std {:.3f}'.format(B2A_mean, B2A_std))

    stat_table = Table(test_config.stat_path)
    print(test_config.stat_path)
    stat_info = OrderedDict({
        'name': test_config.exp_name,
        'ngf': test_config.ngf,
        'ndf': test_config.ndf,
        'n_layers_D': test_config.n_layers_D,
        'n_layers_G': test_config.n_layers_G,
        'G_en_ks': test_config.G_en_ks,
        'G_de_ks': test_config.G_de_ks,
        'D_ks': test_config.D_ks,
        'A2B_mean': A2B_mean,
        'A2B_std': A2B_std,
        'B2_mean': B2A_mean,
        'B2A_std': B2A_std,
    })
    stat_table.write(stat_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-s', '--savepath', type = str, required = False, help = 'chooose  where to save the visualizations')
    parser.add_argument('-e', '--epoch', type = int, default = 99, help = 'choose trained model epoch')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()
    epoch = args.epoch

    config = common.config
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # save_dir = os.path.join('/home1/wurundi/yzh/motion_CycleGAN/test_result', str(epoch))
    tcfg = get_test_config(config, epoch)
    evaluate(tcfg)

