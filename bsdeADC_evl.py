import os

import csv
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from dataloader.infer_dataloader_back import load_multi_model_file, MultiModelMRI, ToTensor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from network.FBSDEGen_CBV import FBSDEGen
from sklearn.model_selection import train_test_split

results_folder = '/home/Sdumt@us21101/data/infer/MT_3modal_newnet'
model_store_path = '/home/Sdumt@us21101/code/BSDE/log/3modal_new_model/bsde_ddp_best.pt'

folder_list = ['/home/Sdumt@us21101/data/MT_test/MT_T1_test',
               '/home/Sdumt@us21101/data/BSDE_T1C',
               '/home/Sdumt@us21101/data/BSDE_ADC',
               '/home/Sdumt@us21101/data/BSDE_CBV',
               '/home/Sdumt@us21101/data/brainmask_1230_new'
               ]


# 转移瘤测试
# MT_test/MT_T1_test
# folder_list = ['/home/Sdumt@us21101/data/GBM_test/GBM_T1_test',
#                '/home/Sdumt@us21101/data/GBM_test/GBM_T1C_test',
#                '/home/Sdumt@us21101/data/GBM_test/GBM_ADC_test',
#                '/home/Sdumt@us21101/data/GBM_test/GBM_CBV_test',
#                '/home/Sdumt@us21101/data/GBM_test/GBM_bm_test']

# device = torch.device('cpu')

cuda_id = 1
device = torch.device(f'cuda:{cuda_id}')
print(f"device={device}")

dim_x = 256
dim_y = 256
dim_z = 32


def calculate_ssim(target, output, mean, thres=255.0):
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    diff_map = output - target

    mae = np.mean(np.abs(diff_map))
    mse = np.mean(np.square(diff_map))
    # psnr = peak_signal_noise_ratio(target, output)

    target = target * mean + mean
    output = output * mean + mean
    # target = torch.clamp(torch.Tensor(target), 0.0, thres).numpy()
    # output = torch.clamp(torch.Tensor(output), 0.0, thres).numpy()
    # assert target.shape == output.shape, "Images must have the same dimensions"
    # assert target.shape == output.shape, "Images must have the same dimensions"

    ncc = np.mean(np.multiply(output - np.mean(output), target - np.mean(target)) / (
            np.std(output) * np.std(target)))

    # ssim_value = ssim(target, output, data_range=thres)
    ssim_value = ssim(target, output, data_range=thres)

    psnr = peak_signal_noise_ratio(target, output,
                                   data_range=max(target.max() - target.min(), output.max() - output.min()))
    return target, output, ssim_value, mae, mse, psnr, ncc


def model_inference(model, multi_dataloader, model_store_path, results_folder, device):
    checkpoint = torch.load(model_store_path, map_location=device)
    # 如果用的是DDP保存的模型的话，需要有这一段
    for key in list(checkpoint.keys()):
        if 'module.y0' in key:
            checkpoint[key.replace('module.y0', 'y0')] = checkpoint[key]
            del checkpoint[key]
        if 'module.z' in key:
            checkpoint[key.replace('module.z', 'z')] = checkpoint[key]
            del checkpoint[key]
        if 'module.ch' in key:
            checkpoint[key.replace('module.ch', 'ch')] = checkpoint[key]
            del checkpoint[key]
    # 不是的话就直接load
    model.load_state_dict(checkpoint)
    model.eval()

    os.makedirs(results_folder, exist_ok=True)

    csv_file_path = os.path.join(results_folder, 'eval.csv')

    with torch.no_grad(), open(csv_file_path, 'w', newline='') as csvfile:

        fieldnames = ['File Name', 'SSIM', 'MAE', 'MSE', 'PSNR', 'NCC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for batch in multi_dataloader:
            input_data = batch['Figure'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)

            output = model(input_data)
            output = output * mask

            # target, output, ssim_eval = calculate_ssim(target[0][0], input_data[0][0])
            # print(ssim_eval)
            # target = target.cpu().numpy()
            # output = output.cpu().numpy()

            # output_data = output.cpu().numpy()

            file_name = batch['file_name']['T1'][0].split(os.sep)[-1][:-8]

            # 转换 spacing
            spacing = [float(s[0]) for s in batch['affine'][0]['spacing']]

            # 转换 origin
            origin = [float(o[0]) for o in batch['affine'][0]['origin']]

            # 转换 direction
            direction = [float(d[0]) for d in batch['affine'][0]['direction']]

            # A, B = np.max(target), np.min(target)
            # a, b = np.max(output_data), np.min(output_data)

            # output_data = ((output_data - b) / (a - b)) * (A - B) + B

            mean = batch['mean'].cpu().numpy()

            # target = target * mean + mean
            # output_data = output_data * mean + mean

            target, output, ssim_eval, mae, mse, psnr, ncc = calculate_ssim(target[0][0], output[0][0], mean)

            print(ssim_eval, mae, mse, psnr, ncc)
            writer.writerow({
                'File Name': file_name,
                'SSIM': ssim_eval,
                'MAE': mae,
                'MSE': mse,
                'PSNR': psnr,
                'NCC': ncc,
            })

            target_image = sitk.GetImageFromArray(np.transpose(target, (2, 1, 0)))
            target_image.SetSpacing(spacing)
            target_image.SetOrigin(origin)
            target_image.SetDirection(direction)

            target_path = os.path.join(results_folder, file_name + "CBV_ori.nii.gz")
            print(target_path)
            sitk.WriteImage(target_image, target_path)

            result_image = sitk.GetImageFromArray(np.transpose(output, (2, 1, 0)))

            result_image.SetSpacing(spacing)
            result_image.SetOrigin(origin)
            result_image.SetDirection(direction)
            result_path = os.path.join(results_folder, file_name + "CBV_sys.nii.gz")
            # result_path = os.path.join(results_folder, file_name[0].split(r'/')[-1])
            print(result_path)
            sitk.WriteImage(result_image, result_path)

# 把模型加载到cuda上
# hyperparameters
T, N = 1.0, 2
dim_w = dim_x

# dim_x, dim_y, dim_w = 32, 784, 32
dim_h1, dim_h2, dim_h3 = 64, 32, 64  # 1000, 600, 1000

# total = load_multi_model_file(folder_list)
#
# train_folders, test_folders = train_test_split(total, test_size=0.3, random_state=3407)
#
# multi_dataset = MultiModelMRI(total, image_shape=(dim_x, dim_y, dim_z))
#
# multi_dataloader = DataLoader(dataset=multi_dataset,
#                               pin_memory=True,
#                               batch_size=1,
#                               num_workers=1, )

total = load_multi_model_file(folder_list)

train_folders, test_folders = train_test_split(total, test_size=0.3, random_state=3407)
multi_dataset = MultiModelMRI(train_folders, image_shape=(dim_x, dim_y, dim_z),
                              transform=transforms.Compose([ToTensor()]))

test_dataset = MultiModelMRI(total, image_shape=(dim_x, dim_y, dim_z),
                             transform=transforms.Compose([ToTensor()]))

multi_dataloader = DataLoader(dataset=multi_dataset,
                              pin_memory=True,
                              shuffle=True,  # 使用分布式训练 shuffle 应该设置为 False
                              batch_size=1,
                              num_workers=1)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

model = FBSDEGen(dim_x, dim_y, dim_h1, dim_h2, dim_h3, T, N, device)
model = model.to(device)

model_inference(model, test_dataloader, model_store_path, results_folder, device)
