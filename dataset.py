import os
import sys
import torch
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import random
import itk

def read_image(fname, imtype):
    reader = itk.ImageFileReader[imtype].New()
    reader.SetFileName(fname)
    reader.Update()
    image = reader.GetOutput()
    return image

def scan_path(d_name, d_path):
    entries = []
    if d_name == 'prostate-3-fold':
        for case_name in os.listdir('{}/image'.format(d_path)):
            if case_name.startswith('Case') and case_name.endswith('.nii.gz'):
                case_id = int(case_name.split('Case')[1].split('.nii.gz')[0])
                image_name = '{0:s}/image/Case{1:04d}.nii.gz'.format(d_path, case_id)
                label_name = '{0:s}/label/Case{1:04d}.nii.gz'.format(d_path, case_id)
                if os.path.isfile(image_name) and os.path.isfile(label_name):
                    entries.append([d_name, 'Case{0:04d}'.format(case_id), image_name, label_name, True])
    return entries

def create_CV_folds(data_path, fraction, exclude_case):
    fold_file_name = '{0:s}/CV-fold.txt'.format(sys.path[0])
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append([params[1], params[2], params[3], params[4], bool(params[5])])
    else:
        entries = []
        for [d_name, d_path] in data_path:
            entries.extend(scan_path(d_name, d_path))
        for e in entries:
            if e[0:2] in exclude_case:
                entries.remove(e)
        random.shuffle(entries)
        ptr = 0
        for fold_id in range(len(fraction)):
            folds[fold_id] = entries[ptr:ptr+fraction[fold_id]]
            ptr += fraction[fold_id]

        with open(fold_file_name, 'w') as fold_file:
            for fold_id in range(len(fraction)):
                for [d_name, case_name, image_path, label_path, labeled] in folds[fold_id]:
                    fold_file.write('{0:d} {1:s} {2:s} {3:s} {4:s} {5:s}\n'.format(fold_id, d_name, case_name, image_path, label_path, str(labeled)))

    folds_size = [len(x) for x in folds.values()]

    return folds, folds_size

def normalize(x, min, max):
    factor = 1.0 / (max - min)
    x[x < min] = min
    x[x > max] = max
    x = (x - min) * factor
    return x

def generate_transform(rand):
    if rand:
        min_rotate = -0.05 # [rad]
        max_rotate = 0.05 # [rad]
        min_offset = -5.0 # [mm]
        max_offset = 5.0 # [mm]
        t = itk.Euler3DTransform[itk.D].New()
        euler_parameters = t.GetParameters()
        euler_parameters = itk.OptimizerParameters[itk.D](t.GetNumberOfParameters())
        offset_x = min_offset + random.random() * (max_offset - min_offset) # rotate
        offset_y = min_offset + random.random() * (max_offset - min_offset) # rotate
        offset_z = min_offset + random.random() * (max_offset - min_offset) # rotate
        rotate_x = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        rotate_y = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        rotate_z = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        euler_parameters[0] = rotate_x # rotate
        euler_parameters[1] = rotate_y # rotate
        euler_parameters[2] = rotate_z # rotate
        euler_parameters[3] = offset_x # tranlate
        euler_parameters[4] = offset_y # tranlate
        euler_parameters[5] = offset_z # tranlate
        t.SetParameters(euler_parameters)
    else:
        offset_x = 0
        offset_y = 0
        offset_z = 0
        rotate_x = 0
        rotate_y = 0
        rotate_z = 0
        t = itk.IdentityTransform[itk.D, 3].New()
    return t, [offset_x, offset_y, offset_z, rotate_x, rotate_y, rotate_z]

def resample(image, imtype, size, spacing, origin, transform, linear, dtype, use_min_default):
    o_origin = image.GetOrigin()
    o_spacing = image.GetSpacing()
    o_size = image.GetBufferedRegion().GetSize()
    output = {}
    output['org_size'] = np.array(o_size, dtype=int)
    output['org_spacing'] = np.array(o_spacing, dtype=np.float32)
    output['org_origin'] = np.array(o_origin, dtype=np.float32)
    
    if origin is None: # if no origin point specified, center align the resampled image with the original image
        new_size = np.zeros(3, dtype=int)
        new_spacing = np.zeros(3, dtype=np.float32)
        new_origin = np.zeros(3, dtype=np.float32)
        for i in range(3):
            new_size[i] = size[i]
            if spacing[i] > 0:
                new_spacing[i] = spacing[i]
                new_origin[i] = o_origin[i] + o_size[i]*o_spacing[i]*0.5 - size[i]*spacing[i]*0.5
            else:
                new_spacing[i] = o_size[i] * o_spacing[i] / size[i]
                new_origin[i] = o_origin[i]
    else:
        new_size = np.array(size, dtype=int)
        new_spacing = np.array(spacing, dtype=np.float32)
        new_origin = np.array(origin, dtype=np.float32)

    output['size'] = new_size
    output['spacing'] = new_spacing
    output['origin'] = new_origin

    resampler = itk.ResampleImageFilter[imtype, imtype].New()
    resampler.SetInput(image)
    resampler.SetSize((int(new_size[0]), int(new_size[1]), int(new_size[2])))
    resampler.SetOutputSpacing((float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])))
    resampler.SetOutputOrigin((float(new_origin[0]), float(new_origin[1]), float(new_origin[2])))
    resampler.SetTransform(transform)
    if linear:
        resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imtype, itk.D].New())
    else:
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imtype, itk.D].New())
    if use_min_default:
        resampler.SetDefaultPixelValue(int(np.min(itk.GetArrayFromImage(image))))
    else:
        resampler.SetDefaultPixelValue(int(np.max(itk.GetArrayFromImage(image))))
    resampler.Update()
    rs_image = resampler.GetOutput()
    image_array = itk.GetArrayFromImage(rs_image)
    image_array = image_array[np.newaxis, :].astype(dtype)
    output['array'] = image_array

    return output

class PolarDataset(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num, aug_data):
        self.ImageType = itk.Image[itk.SS, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.FloatType = itk.Image[itk.F, 3]
        self.ids = ids # list of data samples
        self.rs_size = rs_size # resample image size (in pixel)
        self.rs_spacing = rs_spacing # resample image resolution (in mm)
        self.rs_intensity = rs_intensity # resample image intensity rescaling range
        self.label_map = label_map
        self.cls_num = cls_num
        self.aug_data = aug_data # whether do data augmentation (True or False)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        [d_name, casename, image_fn, label_fn, labeled] = self.ids[index]

        t, t_param = generate_transform(rand=self.aug_data)

        output = {}
        src_image = read_image(fname=image_fn, imtype=self.ImageType)
        image = resample(image=src_image, imtype=self.ImageType, size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                        transform=t, linear=True, dtype=np.float32, use_min_default=True)
        image['array'] = normalize(image['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
        
        src_label = read_image(fname=label_fn, imtype=self.LabelType)
        label = resample(image=src_label, imtype=self.LabelType, size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                    transform=t, linear=False, dtype=np.int64, use_min_default=True)
        tmp_array = np.zeros_like(label['array'])
        lmap = self.label_map[d_name]
        for key in lmap:
            tmp_array[label['array'] == key] = lmap[key]

        # calculate prostate centroid coordinates
        inds = np.nonzero(tmp_array)
        gt_cp = np.zeros_like(image['origin'])
        gt_cp[2] = np.mean(inds[1])
        gt_cp[1] = np.mean(inds[2])
        gt_cp[0] = np.mean(inds[3])
        gt_cp = image['origin'] + gt_cp * image['spacing']

        # augment prostate centroid coordinates
        perturbed_gt_cp = gt_cp.copy()
        if self.aug_data:
            min_offset = -5.0 # [mm]
            max_offset = 5.0 # [mm]
            perturbed_gt_cp[0] += min_offset + random.random() * (max_offset - min_offset)
            perturbed_gt_cp[1] += min_offset + random.random() * (max_offset - min_offset)
            perturbed_gt_cp[2] += min_offset + random.random() * (max_offset - min_offset)
        
        # rescale coordinates to range of [-1, 1]
        gt_cp = ((gt_cp - image['origin']) / ((image['size'] - 1) * image['spacing'])) * 2 - 1
        perturbed_gt_cp = ((perturbed_gt_cp - image['origin']) / ((image['size'] - 1) * image['spacing'])) * 2 - 1

        label['array'] = tmp_array.astype(np.float32)

        output['data'] = torch.from_numpy(image['array'])
        output['label'] = torch.from_numpy(label['array'])
        output['gt_cp'] = torch.from_numpy(gt_cp.astype(np.float32))
        output['perturbed_gt_cp'] = torch.from_numpy(perturbed_gt_cp.astype(np.float32))
        output['dataset'] = d_name
        output['case'] = casename
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['transform'] = np.array(t_param, dtype=np.float32)
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['eof'] = True

        return output