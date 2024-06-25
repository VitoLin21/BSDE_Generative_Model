"""
## Model Training on Single Device (GPU/CPU)
"""
import random
import sys
import time
import os

import argparse

import torch
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from network.FBSDEGen_USCF import FBSDEGen
# from dataloader.dataloader import MultiModelMRI, ToTensor, load_multi_model_file
from dataloader.infer_dataloader_UCSF import MultiModelMRI, ToTensor, load_multi_model_file

from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

logdir = 'log/USCF_ASL_swin_256_0001'

# Setting reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_ssim(target, output, mean, bm, thres=255.0):
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    diff_map = output - target

    mae = np.mean(np.abs(diff_map))
    mse = np.mean(np.square(diff_map))
    psnr = 20 * np.log10(thres) - 10 * np.log10(mse)
    # target = target * mean + mean
    # output = output * mean + mean
    target = (target * mean + mean)
    output = (output * mean + mean)

    target = ((target - target.min()) / (target.max() - target.min()) * 255).astype(np.uint8) * bm
    output = ((output - output.min()) / (output.max() - output.min()) * 255).astype(np.uint8) * bm
    # target = torch.clamp(torch.Tensor(target), 0.0, thres).numpy()
    # output = torch.clamp(torch.Tensor(output), 0.0, thres).numpy()
    assert target.shape == output.shape, "Images must have the same dimensions"

    ncc = np.mean(np.multiply(output - np.mean(output), target - np.mean(target)) / (
            np.std(output) * np.std(target)))

    ssim_value = ssim(target, output, data_range=thres)
    return ssim_value, mae, mse, psnr, ncc


def test(model, dataloader, device):
    model.eval()

    total_ssim = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ncc = 0.0

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input = data['Figure'].to(device)
            target = data['target'].to(device)
            mean = data['mean'].cpu().numpy()
            bm = data['mask'].cpu().numpy()

            y = model(input)

            ssim, mae, mse, psnr, ncc = calculate_ssim(target[0][0], y[0][0], bm[0][0], mean)
            total_ssim += ssim
            total_mae += mae
            total_mse += mse
            total_psnr += psnr
            total_ncc += ncc

    num_samples = len(dataloader)
    avg_ssim = total_ssim / num_samples
    avg_mae = total_mae / num_samples
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ncc = total_ncc / num_samples

    return avg_ssim, avg_mae, avg_mse, avg_psnr, avg_ncc


def train(model, dataloader, test_dataloader, optimizer, n_epochs, store_path, device, local_rank, best_test_path):
    model.train()
    # mse = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    # ssim = SSIM3D(window_size=11)

    writer = SummaryWriter(log_dir=logdir)  # 创建SummaryWriter对象
    best_loss = 1e10
    best_ssim = 1e-10
    best_mae = 1e10
    start_epoch_time = time.time()
    print('start train')
    for epoch in range(n_epochs):
        dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for idx, data in enumerate(dataloader):  # ptr = 0, 5, 10...
            start_time = time.time()  # 记录当前时间
            input = data['Figure'].to(device)
            # print(len(input))
            target = data['target'].to(device)

            optimizer.zero_grad()
            y = model(input)

            # loss = 0.4 * mse(y, target) + 0.6 * ssim(y, target)
            loss = mae_loss(y, target)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"Layer: {name}, Gradient norm: {param.grad.norm()}")
            loss.backward()
            optimizer.step()
            end_time = time.time()  # 记录当前时间
            elapsed_time = end_time - start_time  # 计算经过的时间

            # print(f"epoch={epoch + 1}, Group = {idx}, loss={loss:.8f}, time={elapsed_time:.2f} s")
            # sys.stdout.flush()  # 强制刷新输出缓冲区
            epoch_loss += loss.item()
        end_epoch_time = time.time()
        epoch_elapsed_time = end_epoch_time - start_epoch_time
        epoch_loss /= len(dataloader)
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.8f}"

        # 将信息写入TensorBoardX
        writer.add_scalar('Loss', epoch_loss, epoch)
        # writer.add_scalar('Time', epoch_elapsed_time, epoch)
        if local_rank == 0:
            if best_loss > epoch_loss:
                # Storing the model
                best_loss = epoch_loss
                torch.save(model.state_dict(), store_path)
                log_string += f" --> Best model ({best_loss}) ever (stored)"
                print(log_string)
            else:
                print(log_string)

        # if ((epoch + 1) % 20 == 0) or ((epoch < 50) and ((epoch + 1) % 5 == 0)):
        if (epoch + 1) % 20 == 0:

            _ssim, mae, mse, psnr, ncc = test(model, test_dataloader, device)
            print(f"Test SSIM: {_ssim:.8f}")
            if local_rank == 0:
                if best_ssim < _ssim:
                    # Storing the model
                    best_ssim = _ssim
                    torch.save(model.state_dict(), best_test_path.replace('test', 'test_ssim'))
                    print(f" --> Best model test ssim ({best_ssim}) ever (stored)")

                if best_mae > mae:
                    # Storing the model
                    best_mae = mae
                    torch.save(model.state_dict(), best_test_path.replace('test', 'test_mae'))
                    print(f" --> Best model test mae ({best_mae}) ever (stored)")

            writer.add_scalar('3D SSIM', _ssim, epoch)
            writer.add_scalar('MAE', mae, epoch)
            writer.add_scalar('MSE', mse, epoch)
            writer.add_scalar('PSNR', psnr, epoch)
            writer.add_scalar('NCC', ncc, epoch)
            # 每十个 epoch 保存一次模型
        # if (epoch + 1) % 100 == 0:
        #     save_path = logdir + f"/epoch_{epoch + 1}.pt"
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Model saved at epoch {epoch + 1} as {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--dim_x', default=256, type=int)
    parser.add_argument('--dim_y', default=256, type=int)
    parser.add_argument('--dim_z', default=32, type=int)

    # parser.add_argument('--batch_size', default=20, type=int)

    args = parser.parse_args()
    batch_size = args.batch_size
    n_epochs = args.epoch
    dim_x = args.dim_x
    dim_y = args.dim_y
    dim_z = args.dim_z
    # local_rank = args.local_rank

    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    print(local_rank)
    # 配置每个进程的 GPU， 根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    setup_seed(3407 + local_rank)

    # cuda_id = 1
    # device = torch.device(f'cuda:{cuda_id}')
    # print(f"device={device}")

    store_path = logdir + f"/bsde_ddp_best.pt"
    last_store_path = logdir + f"/bsde_ddp_last.pt"
    best_test_path = logdir + f"/bsde_ddp_best_test.pt"

    folder_list = ['/home/Sdumt@us21101/data/UCSF-PDGM/T1WI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/DWI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/T2WI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/T2FLAIR',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/SWI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/ASL'
                   ]

    # 把模型加载到cuda上
    # hyperparameters
    T, N = 0.1, 2

    # dim_x, dim_y, dim_w = 32, 784, 32
    dim_h1, dim_h2, dim_h3 = 64, 32, 64  # 1000, 600, 1000

    lr = 0.0001

    total = load_multi_model_file(folder_list)
    total = total

    train_folders, others = train_test_split(total, test_size=0.3, random_state=3407)

    val_folders, test_folders = train_test_split(others, test_size=0.33, random_state=3407)

    multi_dataset = MultiModelMRI(train_folders, image_shape=(dim_x, dim_y, dim_z),
                                  transform=transforms.Compose([ToTensor()]))

    val_dataset = MultiModelMRI(val_folders, image_shape=(dim_x, dim_y, dim_z),
                                      transform=transforms.Compose([ToTensor()]))

    test_dataset = MultiModelMRI(test_folders, image_shape=(dim_x, dim_y, dim_z),
                                       transform=transforms.Compose([ToTensor()]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(multi_dataset)
    multi_dataloader = DataLoader(dataset=multi_dataset,
                                  pin_memory=True,
                                  shuffle=(train_sampler is None),  # 使用分布式训练 shuffle 应该设置为 False
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  sampler=train_sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = FBSDEGen(dim_x, dim_y, dim_h1, dim_h2, dim_h3, T, N, device, modal=4, image_shape=(dim_x, dim_y, dim_z))
    model = model.to(device)

    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[local_rank])

    # model = torch.compile(model)

    n_params = sum([p.numel() for p in model.parameters()])
    print(f"number of parameters: {n_params}")

    # try:
    #     model.load_state_dict(torch.load(store_path))
    #     print("#### Model parameters are loaded. ####")
    # except:
    #     pass

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, multi_dataloader, val_dataloader, optimizer, n_epochs, store_path, device, local_rank, best_test_path)

    if local_rank == 0:
        torch.save(model.state_dict(), last_store_path)
    print("Training Completed!")

    print('Start Test')
    _ssim, mae, mse, psnr, ncc = test(model, test_dataloader, device)
    print("_ssim, mae, mse, psnr, ncc:")
    print(_ssim, mae, mse, psnr, ncc)
