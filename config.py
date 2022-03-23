cfg = {}
cfg['model_dir'] = 'model_20220101012345' # folder name where the trained model stored (named in a format of model_YYYYMMDDhhmmss)
cfg['cls_num'] = 1 # number of foreground class (for the prostate ultrasound segmentation cls_num = 1)
cfg['gpu'] = '0' # index of gpu used for training and testing. (to use multiple gpu: cfg['gpu'] = '0,1,2,3')
cfg['fold_num'] = 3 # number of folds used for cross validation
cfg['fold_fraction'] = [105,105,105] # number of samples in each fold
cfg['epoch_num'] = 400 # max training epochs
cfg['batch_size'] = 16 # batch size for training
cfg['test_batch_size'] = 16 # batch size for testing
cfg['lr'] = 0.001 # base learning rate
cfg['model_path'] = '/home/user/proj/prostate/models' # directory where to store the trained models
cfg['im_size'] = 256 # resampled image size (in pixel) (all the original image will be firstly resampled to a size of im_size x im_size x im_size)
cfg['im_spacing'] = 0.5 # resampled image resolution (in millimeter) (all the original image will be firstly resampled to a spacing of im_spacing x im_spacing x im_spacing)
cfg['polar_size'] = [128,64,64] # polar image size: [U, V, R]
cfg['r_spacing'] = 1.0 # resample resolution along the radial direction R in polar transformation. (in millimeter)
cfg['rs_intensity'] = [0.0, 255.0] # rescaled intensity. (from [min, max] to [0, 1])
cfg['cpu_thread'] = 4 # multi-thread for data loading. zero means single thread.
cfg['test_times'] = 20 # inference time K for test-time augmentation (during CPTTA)
cfg['cp_perturb'] = 0.05 # standard deviation sigma of the zero-mean Gaussian noise applied on the predicted centroid coordinates (during CPTTA)


# list of dataset names and paths
cfg['data_path_train'] = [
    ['prostate-3-fold', '/home/user/data/prostate'],
]

# map labels of different datasets to a uniform label map
cfg['label_map'] = {
    'prostate-3-fold':{1:1},
}

# exclude any samples in the form of '[dataset_name, case_name]'
cfg['exclude_case'] = [
]
