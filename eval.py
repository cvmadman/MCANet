# python imports
import os
import glob
from distutils.log import Log
import time
# external imports
import torch
import numpy as np
import torchsnooper
import SimpleITK as sitk
import torch.utils.data as Data
import csv
# internal imports
from network import losses
from network.U_U_DilateAGs import U_UNet, SpatialTransformer
from network import utils
from network.datagenerators import Dataset
#grid
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse

# writer.add_scalar('DSC/train', np.mean(DSC), epoch)
# tensorboard --logdir=Logs
# test时参数
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='./Mind101/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='./Mind101/label')
args = parser.parse_args()

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 208, 160)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[ :,j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:,:, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

    
def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 78:87, :, :]
    fig = plt.figure(figsize=(12,12), dpi=300)
    for i in range(img.shape[0]):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def compute_label_dice(gt, pred, std_idx):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
               1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005,
               2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
               2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035,  11, 12, 13, 17, 50, 51, 52, 53, 16, 24,]

    line = 'p_{}'.format(std_idx)
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
        line = line+','+str(dice)
    return np.mean(dice_lst), line


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

# @torchsnooper.snoop()
def test():
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    test_dir = args.test_dir
    weights = [1, 1, 1.2]
    model_folder = 'U_Utest_ncc_{}_diffusion_{}_NJ_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/' + model_folder[:-1] + '.csv'):
        os.remove('experiments/' + model_folder[:-1] + '.csv')
    if not os.path.exists('experiments/' + model_folder[:-1] ):
        os.makedirs('experiments/' + model_folder[:-1] )
    csv_writter(model_folder[:-1], 'experiments/' + model_folder+ model_folder[:-1])
    line = ''
    for i in range(72):
        line = line + ',' + dict[i]
    csv_writter(line, 'experiments/' + model_folder+ model_folder[:-1])

    log_name = str('U_U_1.4_Dice')
    print("log_name: ", log_name)
    f = open(os.path.join('/Result/', log_name + ".txt"), "a")

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    print("The number of test data: ", len(test_file_lst))
    test_loader = Data.DataLoader(Dataset(files=test_file_lst), batch_size=1, shuffle=False, pin_memory=True,
                                  drop_last=True, num_workers=2)

    img = sitk.ReadImage(args.test_dir+"/64.nii.gz")
    ipt = sitk.GetArrayFromImage(img)[np.newaxis, np.newaxis, ...]
    vol_size = ipt.shape[2:]
    # Set up model
    model = U_UNet(vol_size).to(device)
    checkpoint_path = "./"
    model.load_state_dict(torch.load(checkpoint_path))
    print(checkpoint_path)
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    model.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []
    JAC = []
    JAC_per = []
    step_time = []
    stdy_idx = 0
    for file in test_loader:
        f_name = os.path.split(''.join(file[2]))[1]
        m_name = os.path.split(''.join(file[3]))[1]
        # 读入moving图像
        # moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
        # moving = torch.from_numpy(moving).to(device).float()
        fixed = file[0].to(device).float()
        moving = file[1].to(device).float()
        # 读入fixed、moving图像对应的label
        f_label_file = glob.glob(os.path.join(args.label_dir, f_name[:2] + "*"))[0]
        f_label = sitk.GetArrayFromImage(sitk.ReadImage(f_label_file))[np.newaxis, np.newaxis, ...]
        # f_label = torch.from_numpy(f_label).to(device).float()

        m_label_file = glob.glob(os.path.join(args.label_dir, m_name[:2] + "*"))[0]
        m_label = sitk.GetArrayFromImage(sitk.ReadImage(m_label_file))[np.newaxis, np.newaxis, ...]
        m_label = torch.from_numpy(m_label).to(device).float()
        with torch.no_grad():
            start_time = time.time()
            pred_flow = model(moving, fixed)
            pred_img = STN_img(moving, pred_flow)
            pred_label = STN_label(m_label, pred_flow)
            step_time.append(time.time() - start_time)
            tar = fixed.detach().cpu().numpy()[0, 0, :, :, :]
        #计算雅可比行列式
        jac_det = utils.jacobian_determinant_vxm(pred_flow.detach().cpu().numpy()[0, :, :, :, :])
        print('det<=0: {} -> {}%'.format(np.sum(jac_det <= 0), np.sum(jac_det <= 0) / np.prod(tar.shape)*100))
        JAC.append(np.sum(jac_det < 0))
        JAC_per.append(np.sum(jac_det < 0) / np.prod(tar.shape)*100)
        # 计算DSC
        dice, line = compute_label_dice(f_label, pred_label[0, 0, ...].detach().cpu().numpy(), stdy_idx)
        # line = dice_val_substruct(f_label, pred_label[0, 0, ...].detach().cpu().numpy(), stdy_idx)
        line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
        csv_writter(line, 'experiments/' + model_folder+ model_folder[:-1])
        print("dice: ", dice)
        DSC.append(dice)
        stdy_idx += 1
       
        # if '29Warped' in ''.join(file[2]):
        #     save_image(pred_img, fixed, "29_warped.nii.gz")
        #     save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], fixed, "29_flow.nii.gz")
        #     save_image(pred_label, f_label, "29_label.nii.gz")
        #     grid_img = mk_grid_img(8, 1, (160,208,160))
        #     reg_model_bilin = utils.register_model((160,208,160), 'bilinear')
        #     reg_model_bilin.cuda()
        #     def_grid = reg_model_bilin([grid_img.float(), pred_flow])
        #     grid_fig = comput_fig(def_grid)
        #     plt.savefig("./Result/JPG/AGs_29_grid.jpg")
        #     pred_fig = comput_fig(pred_img)
        #     plt.savefig("./Result/JPG/AGs_29_warped.jpg")
        #     plt.close()
            
        del pred_flow, pred_label

    print('mean(time):',np.mean(step_time)," std(time): ", np.std(step_time)," mean(DSC): ", np.mean(DSC), 
                            "  std(DSC): ", np.std(DSC), "  mean(jac):", np.mean(JAC),"  std(jac):", np.std(JAC),
                            "  mean(%jac):", np.mean(JAC_per),"   std(%jac):",np.std(JAC_per), flush=True)
    print('mean(time):',np.mean(step_time)," std(time): ", np.std(step_time)," mean(DSC): ", np.mean(DSC), 
                            "  std(DSC): ", np.std(DSC), "  mean(jac):", np.mean(JAC),"  std(jac):", np.std(JAC),
                            "  mean(%jac):", np.mean(JAC_per),"   std(%jac):",np.std(JAC_per), file=f) 

if __name__ == "__main__":
    test()
