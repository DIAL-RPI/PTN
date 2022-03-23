import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from dataset import create_CV_folds, PolarDataset
from model import PTN
from utils import polar2file
from metric import eval
from config import cfg

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def train(cfg):
    train_start_time = time.localtime()
    time_stamp = time.strftime("%Y%m%d%H%M%S", train_start_time)
    acc_time = 0
    
    # create directory for results storage
    store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
    loss_fn = '{}/loss.txt'.format(store_dir)
    log_fn = '{}/log.txt'.format(store_dir)
    val_result_path = '{}/results_val'.format(store_dir)
    os.makedirs(val_result_path, exist_ok=True)
    test_result_path = '{}/results_test'.format(store_dir)
    os.makedirs(test_result_path, exist_ok=True)

    # split the dataset into several folds for cross validation
    folds, _ = create_CV_folds(data_path=cfg['data_path_train'], fraction=cfg['fold_fraction'], exclude_case=cfg['exclude_case'])

    # external loop of cross validation
    for fold_id in range(cfg['fold_num']):
        best_model_fn = '{}/fold_{}-epoch_{}.pth.tar'.format(store_dir, fold_id+1, 1)
        fold_start_time = time.localtime()

        # create training fold
        train_val_fold = []
        for i in range(cfg['fold_num'] - 1):
            train_val_fold.extend(folds[(fold_id + i) % cfg['fold_num']])
        train_fold = train_val_fold[:180]
        d_train = PolarDataset(train_fold, rs_size=[cfg['im_size'],cfg['im_size'],cfg['im_size']], rs_spacing=[cfg['im_spacing'],cfg['im_spacing'],cfg['im_spacing']], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=True)
        dl_train = data.DataLoader(dataset=d_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True, drop_last=True, num_workers=cfg['cpu_thread'])
        
        # create validaion fold
        val_fold = train_val_fold[180:]
        d_val = PolarDataset(val_fold, rs_size=[cfg['im_size'],cfg['im_size'],cfg['im_size']], rs_spacing=[cfg['im_spacing'],cfg['im_spacing'],cfg['im_spacing']], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False)
        dl_val = data.DataLoader(dataset=d_val, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        # create testing fold
        test_fold = folds[(fold_id + cfg['fold_num'] - 1) % cfg['fold_num']]
        d_test = PolarDataset(test_fold, rs_size=[cfg['im_size'],cfg['im_size'],cfg['im_size']], rs_spacing=[cfg['im_spacing'],cfg['im_spacing'],cfg['im_spacing']], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False)
        dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        # initialize PTN model for training
        ptn = PTN(in_ch=1, base_ch=32, polar_size=cfg['polar_size'], r_spacing=cfg['r_spacing'], im_size=cfg['im_size'], im_spacing=cfg['im_spacing'])
        ptn = nn.DataParallel(module=ptn)
        ptn.cuda()
        initial_net(ptn)
        ptn.module.loc_fc.weight.data.zero_()
        ptn.module.loc_fc.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

        mse_loss = nn.MSELoss()
        cp_loss = nn.MSELoss()
        optimizer = optim.Adam(ptn.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        best_val_acc = 0.0
        start_epoch = 0

        # print log
        print()
        log_line = "Fold {} of {}:\nTraining settings:\nModel: {}\nModel parameters: {}\nTraining/Validation/Testing samples: {}/{}/{}\nStart time: {}\nConfiguration:\n".format(
            fold_id+1, cfg['fold_num'], ptn.module.description(), sum(x.numel() for x in ptn.parameters()), len(d_train), len(d_val), len(d_test), 
            time.strftime("%Y-%m-%d %H:%M:%S", fold_start_time))
        for cfg_key in cfg:
            log_line += ' --- {}: {}\n'.format(cfg_key, cfg[cfg_key])
        print(log_line)
        with open(log_fn, 'a') as log_file:
            log_file.write(log_line)

        # training + validation loop
        for epoch_id in range(start_epoch, cfg['epoch_num'], 1):

            t0 = time.perf_counter()

            # training
            torch.enable_grad()
            ptn.train()
            epoch_loss = np.zeros(cfg['cls_num'], dtype=np.float)
            epoch_loss_num = np.zeros(cfg['cls_num'], dtype=np.int64)
            batch_id = 0
            for batch in dl_train:

                image = batch['data'].cuda()
                label = batch['label'].cuda()
                gt_cp = batch['gt_cp'].cuda()
                perturbed_gt_cp  = batch['perturbed_gt_cp'].cuda()

                N = len(image)

                pred_srm, pred_cp, gt_srm = ptn(image, gt_cp=perturbed_gt_cp, gt_mask=label)

                print_line = 'Fold {0:d}/{1:d} Epoch {2:d}/{3:d} (train) --- Progress {4:5.2f}% (+{5:02d})'.format(
                    fold_id+1, cfg['fold_num'], epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['batch_size'] / len(d_train), N)

                loss_loc = cp_loss(pred_cp, gt_cp)
                loss_reg = mse_loss(pred_srm, gt_srm)
                loss = loss_loc + loss_reg
                epoch_loss[0] += loss.item()
                epoch_loss_num[0] += 1                        

                print_line += ' --- Loss: {0:.6f}/{1:.6f}/{2:.6f}/({3:.6f},{4:.6f},{5:.6f})-({6:.6f},{7:.6f},{8:.6f})'.format(loss.item(), loss_reg.item(), loss_loc.item(), pred_cp[0,0], pred_cp[0,1], pred_cp[0,2], gt_cp[0,0], gt_cp[0,1], gt_cp[0,2])
                print(print_line)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del image, label, gt_cp, perturbed_gt_cp, pred_cp, gt_srm, pred_srm, loss
                batch_id += 1

            train_loss=np.sum(epoch_loss)/np.sum(epoch_loss_num)
            epoch_loss = epoch_loss / epoch_loss_num

            print_line = 'Fold {0:d}/{1:d} Epoch {2:d}/{3:d} (train) --- Loss: {4:.6f} ({5:s})\n'.format(
                fold_id+1, cfg['fold_num'], epoch_id+1, cfg['epoch_num'], train_loss, '/'.join(['%.6f']*len(epoch_loss)) % tuple(epoch_loss))
            print(print_line)

            # validation
            torch.no_grad()
            ptn.eval()
            for batch_id, batch in enumerate(dl_val):
                image = batch['data']
                N = len(image)

                image = image.cuda()

                pred_srm, pred_cp = ptn(image)

                print_line = 'Fold {0:d}/{1:d} Epoch {2:d}/{3:d} (val) --- Progress {4:5.2f}% (+{5:d})'.format(
                    fold_id+1, cfg['fold_num'], epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['test_batch_size'] / len(d_val), N)
                print(print_line)
                
                for i in range(N):
                    # convert the unit of the prediced centroid point coordinate to millimeter (mm) 
                    pred_cp_in_mm = ((pred_cp[i].detach().cpu() + 1) / 2) * (batch['size'][i] - 1) * batch['spacing'][i] + batch['origin'][i]
                    # reconstruct prostate volume using the predicted surface radius map (SRM) and centroid point
                    polar2file(
                        pred_srm.detach().cpu()[i,:].squeeze() * cfg['polar_size'][2], r_spacing=cfg['r_spacing'], cm=pred_cp_in_mm, 
                        size=batch['org_size'][i], spacing=batch['org_spacing'][i], origin=batch['org_origin'][i], 
                        fn='{}/{}@{}@{}.nii.gz'.format(val_result_path, batch['dataset'][i], batch['case'][i], 1))

                del image, pred_cp, pred_srm
            
            seg_dsc, seg_asd, seg_hd, seg_dsc_m, seg_asd_m, seg_hd_m = eval(
                pd_path=val_result_path, gt_entries=val_fold, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
                metric_fn='metric_{0:d}-{1:04d}'.format(fold_id+1, epoch_id+1), calc_asd=False)
            
            print_line = 'Epoch {0:d}/{1:d} (val) --- DSC {2:.2f} ({3:s})% --- ASD {4:.2f} ({5:s})mm --- HD {6:.2f} ({7:s})mm'.format(
                epoch_id+1, cfg['epoch_num'], 
                seg_dsc_m*100.0, '/'.join(['%.2f']*len(seg_dsc[:,0])) % tuple(seg_dsc[:,0]*100.0), 
                seg_asd_m, '/'.join(['%.2f']*len(seg_asd[:,0])) % tuple(seg_asd[:,0]),
                seg_hd_m, '/'.join(['%.2f']*len(seg_hd[:,0])) % tuple(seg_hd[:,0]))
            print(print_line)

            t1 = time.perf_counter()
            epoch_t = t1 - t0
            acc_time += epoch_t
            print("Epoch time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
                h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

            loss_line = '{fold:>d}\t{epoch:>05d}\t{train_loss:>8.6f}\t{class_loss:s}\t{seg_val_dsc:>8.6f}\t{seg_val_dsc_cls:s}\n'.format(
                fold=fold_id+1, epoch=epoch_id+1, train_loss=train_loss, class_loss='\t'.join(['%8.6f']*len(epoch_loss)) % tuple(epoch_loss), 
                seg_val_dsc=seg_dsc_m, seg_val_dsc_cls='\t'.join(['%8.6f']*len(seg_dsc[:,0])) % tuple(seg_dsc[:,0])
                )
            with open(loss_fn, 'a') as loss_file:
                loss_file.write(loss_line)

            # save best model
            if epoch_id == 0 or seg_dsc_m > best_val_acc:
                # remove former best model
                if os.path.exists(best_model_fn):
                    os.remove(best_model_fn)
                # save current best model
                best_val_acc = seg_dsc_m
                best_model_fn = '{}/fold_{}-epoch_{}.pth.tar'.format(store_dir, fold_id+1, epoch_id+1)                            
                torch.save({
                            'fold':fold_id,
                            'epoch':epoch_id,
                            'acc_time':acc_time,
                            'time_stamp':time_stamp,
                            'fold_start_time':fold_start_time,
                            'best_val_acc':best_val_acc,
                            'best_model_filename':best_model_fn,
                            'model_state_dict':ptn.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()}, 
                            best_model_fn)
                print('Best model (epoch = {}) saved.\n'.format(epoch_id+1))

        # print log
        with open(log_fn, 'a') as log_file:
            log_file.write("Finish time: {finish_time}\nTotal training time: {h:>02d}:{m:>02d}:{s:>02d}\n\n".format(
                    finish_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60))
                        
        # test
        ptn.load_state_dict(torch.load(best_model_fn)['model_state_dict'])
        torch.no_grad()
        ptn.eval()
        for batch_id, batch in enumerate(dl_test):
            image = batch['data']
            N = len(image)

            image = image.cuda()

            pred_srm, pred_cp = ptn(image)

            print_line = 'Testing --- Progress {0:5.2f}% (+{1:d})'.format(100.0 * batch_id * cfg['test_batch_size'] / len(d_test), N)
            print(print_line)

            for i in range(N):
                # convert the unit of the prediced centroid point coordinate to millimeter (mm) 
                pred_cp_in_mm = ((pred_cp[i].detach().cpu() + 1) / 2) * (batch['size'][i] - 1) * batch['spacing'][i] + batch['origin'][i]
                # reconstruct prostate volume using the predicted surface radius map (SRM) and centroid point
                polar2file(
                    pred_srm.detach().cpu()[i,:].squeeze() * cfg['polar_size'][2], r_spacing=cfg['r_spacing'], cm=pred_cp_in_mm, 
                    size=batch['org_size'][i], spacing=batch['org_spacing'][i], origin=batch['org_origin'][i], 
                    fn='{}/{}@{}@{}.nii.gz'.format(test_result_path, batch['dataset'][i], batch['case'][i], 1))

            del image, pred_cp, pred_srm
        
        dsc, asd, hd, dsc_m, asd_m, hd_m = eval(
            pd_path=test_result_path, gt_entries=test_fold, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
            metric_fn='metric_test_fold-{}'.format(fold_id+1), calc_asd=True, keep_largest=True)
        
        print_line = 'Testing fold {0:d} --- DSC {1:.2f} ({2:s})% --- ASD {3:.2f} ({4:s})mm --- HD {5:.2f} ({6:s})mm'.format(
            fold_id+1,
            dsc_m*100.0, '/'.join(['%.2f']*len(dsc[:,0])) % tuple(dsc[:,0]*100.0), 
            asd_m, '/'.join(['%.2f']*len(asd[:,0])) % tuple(asd[:,0]),
            hd_m, '/'.join(['%.2f']*len(hd[:,0])) % tuple(hd[:,0]))
        print(print_line)

    all_folds = []
    for i in range(cfg['fold_num']):
        all_folds.extend(folds[i])
    dsc, asd, hd, dsc_m, asd_m, hd_m = eval(
        pd_path=test_result_path, gt_entries=all_folds, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
        metric_fn='metric_test_all', calc_asd=True, keep_largest=True)
    
    print_line = 'Testing all --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
        dsc_m*100.0, '/'.join(['%.2f']*len(dsc[:,0])) % tuple(dsc[:,0]*100.0), 
        asd_m, '/'.join(['%.2f']*len(asd[:,0])) % tuple(asd[:,0]),
        hd_m, '/'.join(['%.2f']*len(hd[:,0])) % tuple(hd[:,0]))
    print(print_line)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    train(cfg=cfg)