import time
import argparse
import os
import torch
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter

from dataset import get_dataloader
from model import CycleGANModel
from common import config
from utils import TrainClock, cycle
from visualization import plot_motion
from transform import trans_motion3d_inv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config.isTrain = True

    if not os.path.exists('train_log'):
        os.symlink(config.exp_dir, 'train_log')

    # get dataset
    train_loader = get_dataloader("train", batch_size=config.batch_size)
    val_loader = get_dataloader("test", batch_size=config.batch_size)
    val_cycle = cycle(val_loader)
    dataset_size = len(train_loader)
    print('The number of training motions = %d' % (dataset_size * config.batch_size))

    # create tensorboard writer
    train_tb = SummaryWriter(os.path.join(config.log_dir, 'train.events'))
    val_tb = SummaryWriter(os.path.join(config.log_dir, 'val.events'))

    # get model
    net = CycleGANModel(config)
    net.print_networks(True)

    # start training
    clock = TrainClock()
    net.train()

    for e in range(config.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            net.train()
            net.set_input(data)  # unpack data from dataset and apply preprocessing
            net.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            # get loss
            losses_values = net.get_current_losses()

            # update tensorboard
            train_tb.add_scalars('train_loss', losses_values, global_step=clock.step)

            # visualize
            if clock.step % config.visualize_frequency == 0:
                motion_dict = net.infer()
                for k, v in motion_dict.items():
                    phase = 'h' if k[-1] == 'A' else 'nh'
                    motion3d = train_loader.dataset.preprocess_inv(v.detach().cpu().numpy()[0], phase)
                    img = plot_motion(motion3d, phase)
                    train_tb.add_image(k, img, global_step=clock.step)

            pbar.set_description("EPOCH[{}][{}/{}]".format(e, b, len(train_loader)))
            pbar.set_postfix(OrderedDict(losses_values))

            # validation
            if clock.step % config.val_frequency == 0:
                net.eval()
                data = next(val_cycle)
                net.set_input(data)
                net.forward()

                losses_values = net.get_current_losses()
                val_tb.add_scalars('val_loss', losses_values, global_step=clock.step)

                # visualize
                if clock.step % config.visualize_frequency == 0:
                    motion_dict = net.infer()
                    for k, v in motion_dict.items():
                        phase = 'h' if k[-1] == 'A' else 'nh'
                        motion3d = val_loader.dataset.preprocess_inv(v.detach().cpu().numpy()[0], phase)
                        img = plot_motion(motion3d, phase)
                        val_tb.add_image(k, img, global_step=clock.step)

            clock.tick()

        # leraning_rate to tensorboarrd
        lr = net.optimizers[0].param_groups[0]['lr']
        train_tb.add_scalar("learning_rate", lr, global_step=clock.step)

        if clock.epoch % config.save_frequency == 0:
            net.save_networks(epoch=e)

        clock.tock()
        net.update_learning_rate()  # update learning rates at the end of every epoch.


if __name__ == "__main__":
    main()
