import sys

import SimpleITK as sitk
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import shutil


def register(fixed, moving, numberOfBins=48, samplingPercentage=0.10):
    initx = sitk.CenteredTransformInitializer(sitk.Cast(fixed, moving.GetPixelID()), moving, sitk.Euler3DTransform(),
                                              operationMode=sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 50)
    R.SetInitialTransform(initx)
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
    return outTx


def resize_image_itk(itkimage, newSpacing=None, newSize=None, newfactor=None, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    if newSpacing is not None:
        newSpacing = [originSpacing[idx] if newSpacing[idx] is None else newSpacing[idx] for idx in
                      range(len(originSpacing))]
        factor = originSpacing / np.array(newSpacing)
    elif newSize is not None:
        factor = [np.nan if newSize[idx] is None else newSize[idx] / originSize[idx] for idx in range(len(originSize))]
        meanfactor = np.nanmean(factor)
        meanspacing = np.nanmean(
            [np.nan if newSize[idx] is None else originSpacing[idx] for idx in range(len(originSize))])
        factor = [factor[idx] if factor[idx] is not np.nan else originSpacing[idx] / meanspacing * meanfactor for
                  idx in range(len(originSize))]
    else:
        factor = np.array(originSpacing) / originSpacing[2] * newfactor
        # print(factor)
    factor = np.nan_to_num(factor, nan=1.0)
    tempSize = np.asarray(originSize * factor, np.int32).tolist()

    if newSize is None: newSize = tempSize
    newSize = [tempSize[idx] if newSize[idx] is None else newSize[idx] for idx in range(3)]
    tempSpacing = np.asarray(originSpacing / factor, np.int32).tolist()
    if newSpacing is None: newSpacing = tempSpacing
    tempSize = np.maximum(tempSize, newSize).tolist()
    for idx in range(3):
        newSpacing[idx] = tempSpacing[idx] if newSpacing[idx] is None else newSpacing[idx]
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(tempSize)
    # outori = resampler.GetOutputOrigin()
    # resampler.SetOutputOrigin((-250.0, 249.0, outori[2]))
    resampler.SetOutputSpacing(newSpacing)
    # resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(itkimage.GetPixelID())
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    lowerBoundSz = [(tempSize[idx] - newSize[idx]) // 2 for idx in range(len(tempSize))]
    upperBoundSz = [(tempSize[idx] - newSize[idx] - lowerBoundSz[idx]) for idx in range(len(tempSize))]
    itkimgResampled = sitk.Crop(itkimgResampled, lowerBoundaryCropSize=lowerBoundSz, upperBoundaryCropSize=upperBoundSz)
    return itkimgResampled


def get_aug_crops(center, shift, aug_step, aug_count=1, aug_index=(1,), aug_model='sequency'):
    if aug_model == 'random':
        aug_crops = [
            [center[dim] + min(random.randrange(-shift[dim], shift[dim] + aug_step[dim], aug_step[dim]), shift[dim])
             for dim in range(3)] for idx in range(aug_count)]
        count_of_augs = aug_count
    elif aug_model == 'sequency':
        aug_all_crops = [[center[0] + min(idx, shift[0]),
                          center[1] + min(idy, shift[1]),
                          center[2] + min(idz, shift[2]),
                          ] for idx in np.arange(-shift[0], shift[0] + aug_step[0], aug_step[0])
                         for idy in np.arange(-shift[1], shift[1] + aug_step[1], aug_step[1])
                         for idz in np.arange(-shift[2], shift[2] + aug_step[2], aug_step[2])]
        aug_crops = [aug_all_crops[idx % len(aug_all_crops)] for idx in aug_index]
        count_of_augs = len(aug_all_crops)
    else:
        aug_crops = [center]
        count_of_augs = 1
    return aug_crops, count_of_augs


def standard_normalization(IMG, remove_tail=True, divide='mean'):
    data = sitk.GetArrayFromImage(IMG)
    if np.mean(data) < 0.1:
        # print(np.mean(data))
        data = data * 1000000
        IMG = IMG * 1000000
        if np.mean(data) < 0.1:
            data = data * 1000000
            IMG = IMG * 1000000
    data = data - np.min(data)
    IMG = IMG - np.min(data)
    maxvalue, meanvalue, stdvalue = np.max(data), np.mean(data), np.std(data)
    # print(maxvalue, meanvalue, stdvalue)
    data_mask = np.logical_and(data > 0.1 * meanvalue, data < meanvalue + stdvalue * 3.0)
    mean, std = np.mean(data[data_mask]), np.std(data[data_mask])
    if divide == 'mean':
        nIMG = (IMG - mean) / mean
        min = -1
    else:
        nIMG = (IMG - mean) / std
        min = -mean / std
    if remove_tail:
        nIMG = sitk.Cast(sitk.Minimum(nIMG, sitk.Sqrt(sitk.Maximum(nIMG, 0))) - min, sitk.sitkFloat32) / 2.0
    else:
        nIMG = sitk.Cast(nIMG - min, sitk.sitkFloat32) / 2.0
    return nIMG, mean


def load_multi_model_file(folder_list: list):
    # t1, dwi, adc, t2flair, swi, asl = folder_list
    t1, dwi, adc, t2flair, tumor = folder_list
    imdb = []

    files = os.listdir(t1)
    for file in files:
        if file.endswith('T1WI.nii.gz'):
            t1_path = os.path.join(t1, file)
            DWI_path = os.path.join(dwi, file.replace('T1WI', 'DWI'))
            adc_path = os.path.join(adc, file.replace('T1WI', 'ADC'))
            t2flair_path = os.path.join(t2flair, file.replace('T1WI', 'T2FLAIR'))
            tumor_path = os.path.join(tumor, file.replace('T1WI', 'tumor_segmentation'))

            # swi_path = os.path.join(swi, file.replace('T1WI', 'SWI'))
            # asl_path = os.path.join(asl, file.replace('T1WI', 'ASL'))

            # 检查文件是否存在
            if all(os.path.exists(p) for p in
                   [t1_path, DWI_path, adc_path, t2flair_path, tumor_path]):
                imdb.append({'T1': t1_path, 'DWI': DWI_path,
                             'ADC': adc_path, 'T2FLAIR': t2flair_path, 'TUMOR': tumor_path})

    return imdb


class MultiModelMRI(Dataset):
    def __init__(self, imdb, image_shape, transform=None):
        self.imdb = imdb
        self.transform = transform
        self.image_shape = image_shape

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        data = self.data_pre(idx)
        if self.transform:
            data = self.transform(data)
        return data

    def data_pre(self, index, aug_count=1, aug_index=(1,)):
        flnm = self.imdb[index]
        # adjust the size, flnm: image name, dataessamble: all information
        dataessamble = {'Figure': 'SWI'}
        # spacing = [0.75, 0.75, 3.2292]
        list1 = [1, 1, 1]
        lisdwi = [240, 240, 155]
        newsize = self.image_shape
        spacing = list(map(lambda x, y, z: x * y / z, list1, lisdwi, list(newsize)))
        try:
            T1image = resize_image_itk(sitk.ReadImage(flnm['T1']), newSpacing=spacing, newSize=newsize)
            # DWIimage = resize_image_itk(sitk.ReadImage(flnm['DWI']), newSpacing=spacing, newSize=newsize)
            ADCimage = resize_image_itk(sitk.ReadImage(flnm['ADC']), newSpacing=spacing, newSize=newsize)
            T2FLAIRimage = resize_image_itk(sitk.ReadImage(flnm['T2FLAIR']), newSpacing=spacing, newSize=newsize)
            DWIimage = resize_image_itk(sitk.ReadImage(flnm['DWI']), newSpacing=spacing, newSize=newsize)
            TUMOR_mask = resize_image_itk(sitk.ReadImage(flnm['TUMOR']), newSpacing=spacing, newSize=newsize)

            # ASLimage = resize_image_itk(sitk.ReadImage(flnm['ASL']), newSpacing=spacing, newSize=newsize)

        except Exception as e:
            print("error in loading")
            print(flnm)
            sys.exit(e)

        affine = {'spacing': T1image.GetSpacing(), 'origin': T1image.GetOrigin(),
                  'direction': T1image.GetDirection(),
                  'size': T1image.GetSize(),
                  'depth': T1image.GetDepth(), 'dimension': T1image.GetDimension()}
        dataessamble['affine'] = [affine]

        bm = sitk.GetArrayFromImage(DWIimage) > 0

        T1_std_image = standard_normalization(T1image, remove_tail=False, divide='mean')[0] * 1000
        T1_std_image = sitk.Cast(T1_std_image, sitk.sitkInt16)

        # DWI_std_image = standard_normalization(DWIimage, remove_tail=False, divide='mean')[0] * 1000
        # DWI_std_image = sitk.Cast(DWI_std_image, sitk.sitkInt16)

        ADC_std_image = standard_normalization(ADCimage, remove_tail=False, divide='mean')[0] * 1000
        ADC_std_image = sitk.Cast(ADC_std_image, sitk.sitkInt16)

        T2FLAIR_std_image = standard_normalization(T2FLAIRimage, remove_tail=False, divide='mean')[0] * 1000
        T2FLAIR_std_image = sitk.Cast(T2FLAIR_std_image, sitk.sitkInt16)

        DWI_std_image, DWI_mean = standard_normalization(DWIimage, remove_tail=False, divide='mean')
        DWI_std_image = DWI_std_image * 1000
        DWI_std_image = sitk.Cast(DWI_std_image, sitk.sitkInt16)

        # ASL_std_image, ASL_mean = standard_normalization(ASLimage, remove_tail=False, divide='mean')
        # ASL_std_image = ASL_std_image * 1000
        # ASL_std_image = sitk.Cast(ASL_std_image, sitk.sitkInt16)

        # input_data_DWI = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(DWI_std_image)) / 1000.0), 0)
        input_data_T1 = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(T1_std_image)) / 1000.0), 0)
        input_data_ADC = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(ADC_std_image)) / 1000.0), 0)
        input_data_T2FLAIR = np.expand_dims(
            np.transpose(np.float32(sitk.GetArrayFromImage(T2FLAIR_std_image)) / 1000.0), 0)

        input_data_DWI = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(DWI_std_image)) / 1000.0), 0)
        # target_data_ASL = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(ASL_std_image)) / 1000.0), 0)

        tumor_mask = np.expand_dims(np.transpose(sitk.GetArrayFromImage(TUMOR_mask)),0)
        tumor_mask[tumor_mask > 0] = 1

        dataessamble['target'] = input_data_DWI

        try:
            dataessamble['Figure'] = np.concatenate(
                # (input_data_T1, input_data_ADC, input_data_T2FLAIR), 0)
                (input_data_T1, input_data_ADC, input_data_T2FLAIR), 0)

        except Exception as e:
            print(flnm)
            sys.exit(e)

        dataessamble['file_name'] = flnm
        dataessamble['mean'] = DWI_mean
        dataessamble['mask'] = np.expand_dims(np.transpose(bm), 0)
        dataessamble['tumor'] = tumor_mask

        return dataessamble


class ToTensor(object):
    def __call__(self, sample):
        # Convert ndarrays in sample to Tensors
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in
                sample.items()}

if __name__ == '__main__':
    folder_list = ['/home/Sdumt@us21101/data/UCSF-PDGM/T1WI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/DWI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/ADC',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/T2FLAIR',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/tumor'
                   ]

    transform = transforms.Compose([ToTensor(),  # 添加其他可能需要的数据转换操作
                                    ])

    total = load_multi_model_file(folder_list)
    dataset = MultiModelMRI(total, image_shape=(128, 128, 32), transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for batch in dataloader:
        print(batch['Figure'].shape, batch['target'].shape, batch['tumor'].shape)
        # 这里可以加入你的训练或验证逻辑
        # input = batch['Figure']
        # # print(len(input))
        # target = batch['target']
        # print(input.max(), input.std(), target.max(), target.std())
        # print(batch['target'].mean())
        # for i in range(4):
        #     temp = batch['Figure'][0][i]
        #     print(temp.mean())

