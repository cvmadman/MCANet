# ！/usr/bin/env/ pyhton
# -*- coding:utf-8 -*-
# python imports
from torch.utils.tensorboard import SummaryWriter
from asyncio import wrap_future
import time
from itertools import count
import os
import sys
import glob
import warnings
import time
import argparse
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from network import losses
from network.datagenerators import Dataset
from network.U_U_DilateAGs import U_UNet, SpatialTransformer
from network import utils



def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
               1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005,
               2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
               2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035, 11, 12, 13, 17, 50, 51, 52, 53, 16, 24,]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout  #sys.stdout = f 这句代码意思 之后使用print函数 都将内容打印到 f文件中
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def train():
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True

    # 日志文件
    weights = [1, 1, 1.2] # loss weights
    save_dir = 'U_UNet_ncc_{}_diffusion_{}_NJ_{}/'.format(weights[0], weights[1], weights[2])
    make_dirs()
    if not os.path.exists('./Logs/'+save_dir):
        os.makedirs('./Logs/'+save_dir)
    sys.stdout = Logger('./Logs/'+save_dir)
    writer = SummaryWriter(log_dir="./")
    # 读入fixed图像
    img = sitk.ReadImage(args.train_dir+"/10.nii.gz")
    ipt = sitk.GetArrayFromImage(img)[np.newaxis, np.newaxis, ...]
    vol_size = ipt.shape[2:]
    # [B, C, D, W, H]
    input = np.repeat(ipt, args.batch_size, axis=0)
    input = torch.from_numpy(input).to(device).float()

    # 创建配准网络（UNet）和STN
    model = U_UNet(vol_size).to(device)
    # model.load_state_dict(torch.load("Checkpoint_U_U_1.2/100_AGs.pth"))
    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    STN.train()
    STN_label.train()
    # 模型参数个数
    print("model: ", count_parameters(model))
    epoch_start = 0
    max_epoch = 501
    model_dir = 'experiments/' + save_dir
    updated_lr = round(args.lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
    opt = Adam(model.parameters(), lr=updated_lr)
    print("lr:", updated_lr)

    #损失函数设置
    sim_loss_fn = losses.ncc_loss
    grad_loss_fn1 = losses.gradient_loss
    grad_loss_fn2 = losses.DisplacementRegularizer()

    #数据集设置
    train_path = "/home/hpe/Mind101/train"
    train_files = glob.glob(os.path.join(train_path, '*.nii.gz'))
    print("The number of train data: ", len(train_files))
    train_loader = Data.DataLoader(Dataset(files=train_files), batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_files = glob.glob(os.path.join(args.test_dir, '*.nii.gz'))
    test_loader = Data.DataLoader(Dataset(files=test_files), batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  drop_last=True, num_workers=2)
    for epoch in range(epoch_start, max_epoch):
        
        epoch_lr = round(args.lr * np.power(1 - (epoch) / max_epoch, 0.9), 8)
        epoch_step_time = []
        epoch_loss = []
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            step_start_time = time.time()
            input_fixed = data[0]
            input_fixed = input_fixed.to(device).float()
            input_moving = data[1]
            input_moving = input_moving.to(device).float()

            flow = model(input_moving, input_fixed)
            warped_image = STN(input_moving,flow)

            sim_loss = weights[0]*sim_loss_fn(warped_image, input_fixed)
            grad_loss = weights[1]*grad_loss_fn1(flow) + weights[2]*grad_loss_fn2(flow)
            loss = sim_loss + grad_loss
            epoch_loss.append(loss.item())
            loss_all.update(loss.item(), input_moving.numel())

            opt.zero_grad()
            loss.backward()
            opt.step()

            del warped_image
            del flow
            loss = 0
            sim_loss = 0
            grad_loss = 0
            # flip fixed and moving images
            flow = model(input_fixed, input_moving)
            warped_image = STN(input_fixed,flow)
            sim_loss = weights[0]*sim_loss_fn(warped_image, input_moving)
            grad_loss = weights[1]*grad_loss_fn1(flow) + weights[2]*grad_loss_fn2(flow)
            loss = sim_loss + grad_loss
            epoch_loss.append(loss.item())
            loss_all.update(loss.item(), input_moving.numel())
            print("Iter: %d  loss: %f  sim: %f  grad_l2: %f  grad_NJ: %f " % (idx, loss.item(), weights[0] * sim_loss.item(), weights[1]*grad_loss_fn1(flow).item(), weights[2] * grad_loss_fn2(flow).item()), flush=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            #get compute time
            epoch_step_time.append(time.time()-step_start_time)

        # print epoch info
        writer.add_scalar('Loss/all', loss_all.avg, epoch)
        writer.add_scalar('Loss/sim', weights[0] * sim_loss.item(), epoch)
        writer.add_scalar('Loss/L2', weights[1] * grad_loss_fn1(flow).item(), epoch)
        writer.add_scalar('Loss/train', weights[2] * grad_loss_fn2(flow).item(), epoch)
        writer.add_scalar('Loss/lr', epoch_lr, epoch)
        epoch_info = 'Epoch %d/%d  lr: %f' % (epoch, max_epoch, epoch_lr)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        loss_info = 'loss: %.4e' % (np.mean(epoch_loss))
        print(' - '.join((epoch_info, time_info,loss_info)), flush=True)
    
        if epoch % 20 == 0:
            # Save model checkpoint
            save_file_name = os.path.join('./', '%d_AGs.pth' % epoch)
            torch.save(model.state_dict(), save_file_name)

        if epoch % 10 == 0:
            # Save images
            # m_name = str(epoch) + "_m+.nii.gz"
            # m2f_name = str(epoch) + "_m2f+.nii.gz"
            # save_image(input_moving, f_img, m_name)
            # save_image(warped_image, f_img, m2f_name)
            # print("warped images have saved.") 
            step_time =[]
            DSC = []
            with torch.no_grad():
                for file in test_loader:
                    f_name = os.path.split(''.join(file[2]))[1]
                    m_name = os.path.split(''.join(file[3]))[1]
                    # 读入moving图像
                    # moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
                    # moving = torch.from_numpy(moving).to(device).float()
                    fixed = file[0].to(device).float()
                    moving = file[1].to(device).float()
                    # 读入moving图像对应的label
                    f_label_file = glob.glob(os.path.join(args.label_dir, f_name[:2] + "*"))[0]
                    f_label = sitk.GetArrayFromImage(sitk.ReadImage(f_label_file))[np.newaxis, np.newaxis, ...]
                    # f_label = torch.from_numpy(f_label).to(device).float()
                    m_label_file = glob.glob(os.path.join(args.label_dir, m_name[:2] + "*"))[0]
                    m_label = sitk.GetArrayFromImage(sitk.ReadImage(m_label_file))[np.newaxis, np.newaxis, ...]
                    m_label = torch.from_numpy(m_label).to(device).float()

                    model.eval()
                    start_time = time.time()
                    pred_flow = model(moving, fixed)
                    pred_label = STN_label(m_label, pred_flow)
                    step_time.append(time.time() - start_time)
                    # 计算DSC
                    dice = compute_label_dice(f_label, pred_label[0, 0, ...].detach().cpu().numpy())
                    print("dice: ", dice)
                    DSC.append(dice)
                print('mean(time):',np.mean(step_time), "   mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC), flush=True)
            writer.add_scalar('DSC/train', np.mean(DSC), epoch)
            writer.add_scalar('DSC/time', np.mean(step_time), epoch)


parser = argparse.ArgumentParser()
# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')
# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="/home/hpe/Mind101/train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1)  # recommend 1.0 for ncc, 0.02 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Logs')
# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='/home/hpe/Mind101/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='')
args = parser.parse_args()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
