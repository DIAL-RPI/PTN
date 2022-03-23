import numpy as np
# Note:
# Use itk here will cause deadlock after the first training epoch 
# when using multithread (dataloader num_workers > 0) but reason unknown
import SimpleITK as sitk

import torch
import torch.nn.functional as F
import vtk
from vtk.util import numpy_support
from vtkmodules.util import vtkImageExportToArray
    
def read_image(fname):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fname)
    image = reader.Execute()
    return image

def resample_array(array, size, spacing, origin, size_rs, spacing_rs, origin_rs, transform=None, linear=False):
    array = np.reshape(array, [size[2], size[1], size[0]])
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size_rs[0]), int(size_rs[1]), int(size_rs[2])))
    resampler.SetOutputSpacing((float(spacing_rs[0]), float(spacing_rs[1]), float(spacing_rs[2])))
    resampler.SetOutputOrigin((float(origin_rs[0]), float(origin_rs[1]), float(origin_rs[2])))
    if transform is not None:
        resampler.SetTransform(transform)
    else:
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if linear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    rs_image = resampler.Execute(image)
    rs_array = sitk.GetArrayFromImage(rs_image)

    return rs_array

def polar2file(grid_r, r_spacing, cm, size, spacing, origin, fn, transform=None):
    [V, U] = grid_r.shape
    u = torch.linspace(0, 360, steps=U) * np.pi / 180
    v = torch.linspace(-90, 90, steps=V) * np.pi / 180
    grid_v, grid_u = torch.meshgrid(v, u)
    grid_x = (grid_r - 1) * r_spacing * torch.cos(grid_v) * torch.sin(grid_u) + cm[0]
    grid_y = (grid_r - 1) * r_spacing * torch.cos(grid_v) * torch.cos(grid_u) + cm[1]
    grid_z = (grid_r - 1) * r_spacing * torch.sin(grid_v) + cm[2]
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    grid_z = grid_z.unsqueeze(0)
    grid = torch.cat([grid_x, grid_y, grid_z], dim=0)
    grid = grid.view(3, -1).permute(1, 0)
    grid_arr = grid.numpy()

    verts = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    pd = vtk.vtkPolyData()
    verts.SetNumberOfPoints(grid_arr.shape[0])
    verts.SetData(numpy_support.numpy_to_vtk(grid_arr))

    cells_grid_x = np.linspace(0, grid_r.shape[1]-1, grid_r.shape[1], dtype=np.int64)
    cells_grid_y = np.linspace(0, grid_r.shape[0]-1, grid_r.shape[0], dtype=np.int64)
    cells_grid_x, cells_grid_y = np.meshgrid(cells_grid_x, cells_grid_y)

    cells_npy_0 = cells_grid_y[0:grid_r.shape[0]-1,:] * grid_r.shape[1] + cells_grid_x[0:grid_r.shape[0]-1,:]
    cells_npy_1 = cells_grid_y[0:grid_r.shape[0]-1,:] * grid_r.shape[1] + ((cells_grid_x[0:grid_r.shape[0]-1,:] + 1) % grid_r.shape[1])
    cells_npy_2 = cells_grid_y[1:grid_r.shape[0],:] * grid_r.shape[1] + cells_grid_x[0:grid_r.shape[0]-1,:]
    cells_npy_3 = cells_grid_y[1:grid_r.shape[0],:] * grid_r.shape[1] + ((cells_grid_x[0:grid_r.shape[0]-1,:] + 1) % grid_r.shape[1])

    cells_npy_0 = cells_npy_0.flatten()
    cells_npy_1 = cells_npy_1.flatten()
    cells_npy_2 = cells_npy_2.flatten()
    cells_npy_3 = cells_npy_3.flatten()
    cells_npy = np.hstack([
        np.vstack([np.ones(len(cells_npy_0), dtype=np.int64) * 3, cells_npy_0, cells_npy_1, cells_npy_2]), 
        np.vstack([np.ones(len(cells_npy_0), dtype=np.int64) * 3, cells_npy_1, cells_npy_2, cells_npy_3])
        ]).T.flatten()
    cells.SetNumberOfCells(len(cells_npy_0) * 2)
    cells.SetCells(len(cells_npy_0) * 2, numpy_support.numpy_to_vtkIdTypeArray(cells_npy))

    pd.SetPoints(verts)
    pd.SetPolys(cells)
    pd.Modified()

    white_image = vtk.vtkImageData()
    white_image.SetSpacing((spacing[0], spacing[1], spacing[2]))
    white_image.SetOrigin((origin[0], origin[1], origin[2]))
    white_image.SetDimensions((size[0], size[1], size[2]))
    white_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    white_image.GetPointData().GetScalars().Fill(1)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputOrigin(origin[0], origin[1], origin[2])
    pol2stenc.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    v2a = vtkImageExportToArray.vtkImageExportToArray()
    v2a.SetInputData(imgstenc.GetOutput())
    array = v2a.GetArray()

    if transform is not None:
        array = resample_array(array, size, spacing, origin, size, spacing, origin, transform=transform, linear=False)

    label = sitk.GetImageFromArray(array)
    label.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    label.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    lb_writer = sitk.ImageFileWriter()
    lb_writer.SetFileName(fn)
    lb_writer.Execute(label)

    rmap = sitk.GetImageFromArray(grid_r.numpy())
    rmap_writer = sitk.ImageFileWriter()
    rmap_writer.SetFileName('{}-rmap.nii.gz'.format(fn))
    rmap_writer.Execute(rmap)