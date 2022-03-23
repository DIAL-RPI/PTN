import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from dataset import create_CV_folds, PolarDataset
from model import PTN
from utils import polar2file
from metric import eval
import SimpleITK as sitk
from config import cfg

def get_best_model_name(dir, fold_id):
    for fn in os.listdir(dir):
        if fn.startswith('fold_{0:d}-epoch_'.format(fold_id)) and fn.endswith('.pth.tar'):
            return fn
    return ''

def test(cfg):
    # create directory for results storage
    loc_store_dir = '{}/{}'.format(cfg['model_path'], cfg['model_dir'])

    # split the dataset into several folds for cross validation
    folds, _ = create_CV_folds(data_path=cfg['data_path_train'], fraction=cfg['fold_fraction'], exclude_case=cfg['exclude_case'])

    # external loop of cross validation
    for fold_id in range(cfg['fold_num']):
        # find the model file stored for this fold
        best_loc_model_fn = '{}/{}'.format(loc_store_dir, get_best_model_name(loc_store_dir, fold_id+1))

        # create testing fold
        test_fold = folds[(fold_id + cfg['fold_num'] - 1) % cfg['fold_num']]
        d_test = PolarDataset(test_fold, rs_size=[cfg['im_size'],cfg['im_size'],cfg['im_size']], rs_spacing=[cfg['im_spacing'],cfg['im_spacing'],cfg['im_spacing']], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False)
        dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        # load the trained model
        ptn = PTN(in_ch=1, base_ch=32, polar_size=cfg['polar_size'], r_spacing=cfg['r_spacing'], im_size=cfg['im_size'], im_spacing=cfg['im_spacing'])
        ptn = nn.DataParallel(module=ptn)
        ptn.cuda()
        ptn.load_state_dict(torch.load(best_loc_model_fn)['model_state_dict'])

        # test
        torch.no_grad()
        ptn.eval()
        for batch_id, batch in enumerate(dl_test):
            image = batch['data'].cuda()
            N = len(image)

            # get the predicted prostate centroid coordinates
            _, pred_cp_tmp = ptn(image)
            pred_cp = pred_cp_tmp.detach()
            del pred_cp_tmp

            # loop of multiple inferences for centroid perturbed test-time augmentation (CPTTA)
            for test_iter in range(cfg['test_times']):
                # impose a random perturbation on the predicted prostate centroid coordinate
                noisy_cp = pred_cp + cfg['cp_perturb'] * torch.randn_like(pred_cp)

                # make prediction using the input image and the perturbed centroid point
                pred_srm, _ = ptn(image, lb_cp=noisy_cp)

                print_line = 'Testing --- Progress {0:5.2f}% (+{1:d})'.format(100.0 * batch_id * cfg['test_batch_size'] / len(d_test), N)
                print(print_line)

                test_result_path = '{0:s}/tta-norm-{1:.2f}/t-{2:d}'.format(loc_store_dir, cfg['cp_perturb'], test_iter)
                os.makedirs(test_result_path, exist_ok=True)

                for i in range(N):
                    # convert the unit of the prediced centroid point coordinate to millimeter (mm) 
                    pred_cp_in_mm = ((noisy_cp[i].detach().cpu() + 1) / 2) * (batch['size'][i] - 1) * batch['spacing'][i] + batch['origin'][i]
                    # reconstruct prostate volume using the predicted surface radius map (SRM) and centroid point
                    polar2file(
                        pred_srm.detach().cpu()[i,:].squeeze() * cfg['polar_size'][2], r_spacing=cfg['r_spacing'], cm=pred_cp_in_mm, 
                        size=batch['org_size'][i], spacing=batch['org_spacing'][i], origin=batch['org_origin'][i], 
                        fn='{}/{}@{}@{}.nii.gz'.format(test_result_path, batch['dataset'][i], batch['case'][i], 1))

                del noisy_cp, pred_srm

            del image, pred_cp
    
    all_folds = []
    for i in range(cfg['fold_num']):
        all_folds.extend(folds[i])
        
    # calculate mean and standard deviation (uncertainty) for each case
    for [d_name, casename, _, _, _] in all_folds:
        
        mean_arr = None
        mean2_arr = None
        final_result_path = '{0:s}/tta-norm-{1:.2f}/_avg'.format(loc_store_dir, cfg['cp_perturb'])
        os.makedirs(final_result_path, exist_ok=True)
        for test_iter in range(cfg['test_times']):

            test_result_path = '{0:s}/tta-norm-{1:.2f}/t-{2:d}'.format(loc_store_dir, cfg['cp_perturb'], test_iter)
            tta_fname = '{}/{}@{}@1.nii.gz'.format(test_result_path, d_name, casename)
            
            reader = sitk.ImageFileReader()
            reader.SetFileName(tta_fname)
            tta_image = reader.Execute()
            tta_array = sitk.GetArrayFromImage(tta_image).astype(dtype=np.float)
            if mean_arr is None:
                mean_arr = tta_array
                mean2_arr = tta_array**2
            else:
                mean_arr += tta_array 
                mean2_arr += tta_array**2

        mean_arr = mean_arr / cfg['test_times']
        mean2_arr = mean2_arr / cfg['test_times']
        std_arr = np.sqrt(mean2_arr - mean_arr**2)

        bin_arr = np.zeros_like(mean_arr).astype(dtype=np.uint8)
        bin_arr[mean_arr > 0.5] = 1
        pd_image = sitk.GetImageFromArray(bin_arr)
        pd_image.SetOrigin(tta_image.GetOrigin())
        pd_image.SetSpacing(tta_image.GetSpacing())
        writer = sitk.ImageFileWriter()
        writer.SetFileName('{}/{}@{}@1.nii.gz'.format(final_result_path, d_name, casename))
        writer.Execute(pd_image)
        pd_uncertainty = sitk.GetImageFromArray(std_arr)
        pd_uncertainty.SetOrigin(tta_image.GetOrigin())
        pd_uncertainty.SetSpacing(tta_image.GetSpacing())
        writer = sitk.ImageFileWriter()
        writer.SetFileName('{}/{}@{}@1-uncertainty.nii.gz'.format(final_result_path, d_name, casename))
        writer.Execute(pd_uncertainty)

    # calculate metrics (DSC/ASD/HD) over all cases
    dsc, asd, hd, dsc_m, asd_m, hd_m = eval(
        pd_path=final_result_path, gt_entries=all_folds, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
        metric_fn='metric_test_all', calc_asd=True, keep_largest=True)
    
    print_line = 'Testing all --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
        dsc_m*100.0, '/'.join(['%.2f']*len(dsc[:,0])) % tuple(dsc[:,0]*100.0), 
        asd_m, '/'.join(['%.2f']*len(asd[:,0])) % tuple(asd[:,0]),
        hd_m, '/'.join(['%.2f']*len(hd[:,0])) % tuple(hd[:,0]))
    print(print_line)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    test(cfg=cfg)