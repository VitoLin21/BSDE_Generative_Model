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
        if np.mean(data) < 0.1:
            data = data * 1000000

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
    return nIMG


def load_multi_model_file(folder_list: list):
    t1, t1c, adc, t2wi, t2flair = folder_list

    imdb = []

    for root, dirs, files in os.walk(t1):
        for file in files:
            if file.endswith('T1WI.nii.gz'):
                head = file[:-11]
                t1_path = os.path.join(t1, file)
                t1c_path = os.path.join(t1c, head + 'T1C.nii.gz')
                adc_path = os.path.join(adc, head + 'ADC.nii.gz')
                t2wi_path = os.path.join(t2wi, head + 'T2WI.nii.gz')
                t2flair_path = os.path.join(t2flair, head + 'T2FLAIR.nii.gz')

                # 检查文件是否存在
                if all(os.path.exists(p) for p in
                       [t1_path, t1c_path, adc_path, t2wi_path, t2flair_path]):
                    imdb.append({'T1': t1_path, 'T1C': t1c_path, 'ADC': adc_path,
                                 'T2WI': t2wi_path, 'T2FLAIR': t2flair_path})
                    #
                    person_folder = os.path.join(os.path.dirname(t1), 'person', head)
                    os.makedirs(person_folder, exist_ok=True)
                    for path in [t1_path, t1c_path, adc_path, t2wi_path, t2flair_path]:
                        shutil.copy(path, os.path.join(person_folder, path.split(os.sep)[-1]))
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
        dataessamble = {'Figure': 'ADC'}
        # spacing = [0.75, 0.75, 3.2292]
        spacing = [1.5, 1.5, 4.84375]
        newsize = self.image_shape
        try:
            T1Cimage = resize_image_itk(sitk.ReadImage(flnm['T1C']), newSpacing=spacing, newSize=newsize)
            T1image = resize_image_itk(sitk.ReadImage(flnm['T1']), newSpacing=spacing, newSize=newsize)
            T2WIimage = resize_image_itk(sitk.ReadImage(flnm['T2WI']), newSpacing=spacing, newSize=newsize)
            T2FLAIRimage = resize_image_itk(sitk.ReadImage(flnm['T2FLAIR']), newSpacing=spacing, newSize=newsize)
            ADCimage = resize_image_itk(sitk.ReadImage(flnm['ADC']), newSpacing=spacing, newSize=newsize)
        except Exception as e:
            print("error in loading T1C and bm")
            print(flnm)
            sys.exit(e)

        affine = {'spacing': T1Cimage.GetSpacing(), 'origin': T1Cimage.GetOrigin(),
                  'direction': T1Cimage.GetDirection(),
                  'size': T1Cimage.GetSize(),
                  'depth': T1Cimage.GetDepth(), 'dimension': T1Cimage.GetDimension()}
        dataessamble['affine'] = affine


        # print(flnm['T1'])
        # T1Cimage = sitk.Cast(T1Cimage, sitk.sitkInt16)
        T1C_std_image = standard_normalization(T1Cimage, remove_tail=False, divide='mean') * 1000
        T1C_std_image = sitk.Cast(T1C_std_image, sitk.sitkInt16)

        # T1image = sitk.Cast(T1image, sitk.sitkInt16)
        T1_std_image = standard_normalization(T1image, remove_tail=False, divide='mean') * 1000
        T1_std_image = sitk.Cast(T1_std_image, sitk.sitkInt16)

        # T2WIimage = sitk.Cast(T2WIimage, sitk.sitkInt16)
        T2WI_std_image = standard_normalization(T2WIimage, remove_tail=False, divide='mean') * 1000
        T2WI_std_image = sitk.Cast(T2WI_std_image, sitk.sitkInt16)

        # T2FLAIRimage = sitk.Cast(T2FLAIRimage, sitk.sitkInt16)
        T2FLAIR_std_image = standard_normalization(T2FLAIRimage, remove_tail=False, divide='mean') * 1000 
        T2FLAIR_std_image = sitk.Cast(T2FLAIR_std_image, sitk.sitkInt16)

        # ADCimage = sitk.Cast(ADCimage, sitk.sitkInt16)

        ADC_std_image = standard_normalization(ADCimage, remove_tail=False, divide='mean') * 1000 
        ADC_std_image = sitk.Cast(ADC_std_image, sitk.sitkInt16)

        # sitk.WriteImage(ADCimage, os.path.join('/home/Sdumt@us21201/data/500/test',
        #                                        flnm['ADC'].split(os.sep)[-1][:-7] + '_norm.nii.gz'))

        # print('finish')

        input_data_T1C = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(T1C_std_image)) / 1000.0), 0)
        bm = input_data_T1C > 0

        input_data_ADC = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(ADC_std_image)) / 1000.0), 0) * bm
        input_data_T1 = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(T1_std_image)) / 1000.0), 0) * bm
        input_data_T2WI = np.expand_dims(np.transpose(np.float32(sitk.GetArrayFromImage(T2WI_std_image)) / 1000.0), 0) * bm
        input_data_T2FLAIR = np.expand_dims(
            np.transpose(np.float32(sitk.GetArrayFromImage(T2FLAIR_std_image)) / 1000.0), 0) * bm


        dataessamble['mask'] = bm

        try:
            # dataessamble['Figure'] = np.concatenate(
            #     (input_data_T1, input_data_T1C, input_data_ADC), 0)
            dataessamble['Figure'] = np.concatenate(
              (input_data_T1, input_data_T1C, input_data_ADC, input_data_T2WI, input_data_T2FLAIR), 0)
        except Exception as e:
            print(flnm)
            sys.exit(e)

        dataessamble['root'] = flnm

        # data augmentation
        refdata = dataessamble['Figure']
        aug_side = (16, 16, 16)
        aug_stride = (8, 8, 8)
        aug_step = np.maximum(aug_stride, 1)
        image_size = np.shape(refdata)[1], np.shape(refdata)[2], np.shape(refdata)[3]
        data_shape = self.image_shape
        center_shift = (0, 0, 0)

        aug_range = [min(aug_side[dim], (image_size[dim] - data_shape[dim] - center_shift[dim]) // 2) for dim in
                     range(3)]
        aug_center = [(image_size[dim] + center_shift[dim] - data_shape[dim]) // 2 for dim in range(3)]
        aug_model = 'center'
        aug_crops, count_of_augs = get_aug_crops(aug_center, aug_range, aug_step,
                                                 aug_count=aug_count, aug_index=aug_index, aug_model=aug_model)
        aug_crops = [[sX1, sY1, sZ1, sX1 + data_shape[0], sY1 + data_shape[1], sZ1 + data_shape[2]] for sX1, sY1, sZ1 in
                     aug_crops]  # 是增强区域的坐标列表

        data = {'orig_size': dataessamble['affine']['size'],
                'aug_crops': aug_crops,
                'count_of_augs': [count_of_augs],
                'affine': [dataessamble['affine']],
                'Figure': [dataessamble['Figure'][:, sX1:sX2, sY1:sY2, sZ1:sZ2]
                           for sX1, sY1, sZ1, sX2, sY2, sZ2 in aug_crops][0],
                # 'target': [dataessamble['target'][:, sX1:sX2, sY1:sY2, sZ1:sZ2]
                #            for sX1, sY1, sZ1, sX2, sY2, sZ2 in aug_crops][0],
                'mask': [dataessamble['mask'][:, sX1:sX2, sY1:sY2, sZ1:sZ2]
                         for sX1, sY1, sZ1, sX2, sY2, sZ2 in aug_crops][0],
                'file_name': flnm}
        # datainput['Figure'] = np.squeeze(datainput['Figure'], axis=4)
        # datainput['Figure'] = np.expand_dims(datainput['Figure'], axis=0)
        # datainput['label'] = dataessamble['label'][np.newaxis, :]
        return data


class ToTensor(object):
    def __call__(self, sample):
        # Convert ndarrays in sample to Tensors
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in
                sample.items()}


if __name__ == '__main__':
    folder_list = ['/home/Sdumt@us21101/data/UCSF-PDGM/T1WI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/T1C',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/ADC',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/T2WI',
                   '/home/Sdumt@us21101/data/UCSF-PDGM/T2FLAIR',
                   ]

    # imdb_ADC_train = load_multi_model_file(folder_list)
    # # imdb_CBV_train = load_file(folder_path_CBV_train)
    # #
    # image_shape = [224, 224, 3]
    # #
    # flnmCs, inputCs = readbatch(imdb_ADC_train, range(5), image_shape)
    # # TflnmCs, TinputCs = readbatch(imdb_CBV_train, range(0, 5), image_shape)
    #
    # print(imdb_ADC_train)

    transform = transforms.Compose([ToTensor(),  # 添加其他可能需要的数据转换操作
                                    ])

    total = load_multi_model_file(folder_list)
    dataset = MultiModelMRI(total, image_shape=(128, 128, 64), transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for batch in dataloader:
        # 这里可以加入你的训练或验证逻辑
        print(batch['file_name'])