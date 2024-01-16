# coding:utf-8
import os
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.my_dataset import My_dataset
from util.util import calculate_accuracy,intersect_and_union,f_score, intersect_and_union1,calculate_accuracy1
from tqdm import tqdm
from tensorboardX import SummaryWriter
from train_parameters import n_class,in_wh,model_name
from train_parameters import train_map,train_label,val_map,val_label
from train_parameters import map_seffix,label_seffix
from train_parameters import augmentation_methods

from train_parameters import project_name,model_name,in_chans,out_chans
from train_parameters import batch_size,epoch_max,epoch_from,num_workers
from train_parameters import lr_start,lr_decay,opt_name,loss_name
from train_parameters import gpu
from train_parameters import model_dir,tensorboard_log_dir
from train_parameters import continue_checkpoint_optim_file_name,continue_checkpoint_model_file_name

from loss.lossfunction import CrossEntropy,FocalLoss,BinaryCrossEntropy,SoftIoULoss
import model
import math
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
def train(epo, model, train_loader, optimizer):
   # if epo<=10:
        #lr_this_epo = (lr_start-1e-6)/10*epo + 1e-6
   # else:
       # lr_this_epo = 1e-6 + 0.5*(lr_start-1e-6)*(1 + math.cos(math.pi * epo / (epoch_max)))
    # lr_this_epo = lr_start*lr_decay**epo
    loss_avg = 0.
    acc_avg  = 0.
    iou_avg =  0.
    total_area_intersect = 0.
    total_area_union = 0.
    start_t = t = time.time()
    model.train()
    scaler = GradScaler()
    for it, (images, labels, names) in enumerate(train_loader):
        if (epo-1)*len(train_loader) + it < 10*len(train_loader):
            lr_this_epo = (lr_start-5e-6)/10/len(train_loader)* ((epo-1)*len(train_loader) + it ) + 5e-6
        else:
            lr_this_epo = 5e-6 + lr_start*(1 + math.cos( math.pi*((epo-1)*len(train_loader) + it )/epoch_max/len(train_loader)  ))/2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo              
        # images = Variable(images).cuda(gpu) 
        # labels = Variable(labels).cuda(gpu)
        if gpu >= 0:
            images = images.cuda(gpu)
            images = images.float()
            labels = labels.cuda(gpu)
        optimizer.zero_grad()

        with autocast():

            logits = model(images)
            logits = logits.squeeze(1)
            loss = eval(loss_name)(logits=logits,labels=labels.float())
            loss = loss.loss_function()
       
            iouloss = SoftIoULoss(1)
            iouloss = iouloss(logits,labels)
            loss = 0.1*iouloss + loss  #+  loss2 +  loss3 + loss0 + loss1#+ 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        # acc_avg  += float(acc)
        area_intersect, area_union,_,_ = intersect_and_union1(2, logits, labels)
        iou = area_intersect[1] / (area_union[1] + 1e-6)
        total_area_intersect += area_intersect[1]
        total_area_union += area_union[1]
        # iou_avg += iou[1]
        cur_t = time.time()
        if cur_t-t > 5:
            print('learning rate: %.9f ' % lr_this_epo)
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, iou: %.4f' \
                % (epo, epoch_max, it+1, train_loader.n_iter, (it+1)*batch_size/(cur_t-start_t), float(loss), float(iou)))
            t += 5

    content = '| epo:%s/%s \nlr:%.4f train_loss_avg:%.4f train_iou:%.4f ' \
            % (epo, epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, total_area_intersect/total_area_union)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content+'\n')
    return format(total_area_intersect/total_area_union,'.4f'),format(loss_avg/train_loader.n_iter,'.4f')


def validation(epo, model, val_loader):

    loss_avg = 0.
    acc_avg  = 0.
    iou_avg  = 0.
    start_t = t= time.time()
    model.eval()
    
    total_area_intersect = torch.zeros((n_class, ), dtype=torch.float64)
    total_area_union = torch.zeros((n_class, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((n_class, ), dtype=torch.float64)
    total_area_label = torch.zeros((n_class, ), dtype=torch.float64)

    total_area_intersect = total_area_intersect.cuda(gpu)
    total_area_union = total_area_union.cuda(gpu)
    total_area_pred_label = total_area_pred_label.cuda(gpu)
    total_area_label = total_area_label.cuda(gpu)
    
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images)
            labels = Variable(labels)
            if gpu >= 0:
                images = images.cuda(gpu)
                images = images.float()
                labels = labels.cuda(gpu)
            logits = model(images)
            logits = logits.squeeze(1)
            loss =  eval(loss_name)(logits=logits,labels=labels.float())
            loss = loss.loss_function()
            acc = calculate_accuracy1(logits, labels)
            # for i in range(logits.shape[0]):
            #     it_logit = logits[i]
            #     it_label = labels[i]
            #     area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            #         n_class, it_logit, it_label)
            #     total_area_intersect += area_intersect
            #     total_area_union += area_union
            #     total_area_pred_label += area_pred_label
            #     total_area_label += area_label
            
            area_intersect, area_union, area_pred_label, area_label = intersect_and_union1(
                n_class, logits, labels)
            iou = area_intersect[1]/(area_union[1]+1e-6)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
            loss_avg += float(loss)
            acc_avg  += float(acc)

            cur_t = time.time()
            if cur_t-t > 5:
                print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, iou: %.4f' \
                    % (epo, epoch_max, it+1, val_loader.n_iter, (it+1)*batch_size/(cur_t-start_t), float(loss), float(iou)))
                t += 5
    
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label


    # fwiou = total_area_label/(sum(total_area_union + total_area_intersect)/2) * iou
    # fwiou = sum(fwiou)


    beta = 1
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
    dice = 2 * total_area_intersect / (
        total_area_pred_label + total_area_label)
    acc = total_area_intersect / total_area_label




    mtx1 = '| val_loss_avg:%.4f val_acc_avg:%.4f iou:%.4f\n' \
        % (loss_avg/val_loader.n_iter, acc_avg/val_loader.n_iter,iou[1])
    mtx2 = '|all_acc:'+str(all_acc.cpu().numpy())+'\n'
    mtx3 = '|IoU'+str(iou.cpu().numpy())+'\n'
    mtx4 = '|Acc'+str(acc.cpu().numpy())+'\n'
    mtx5 = '|Fscore'+str(f_value.cpu().numpy())+'\n'
    mtx6 = '|Precision'+str(precision.cpu().numpy())+'\n'
    mtx7 = '|Recall'+str(recall.cpu().numpy())+'\n'
    mtx8 = '|Dice'+str(dice.cpu().numpy())+'\n'
    
    print(mtx1,mtx2,mtx3,mtx4,mtx5,mtx6,mtx7,mtx8)
        
    
    with open(log_file, 'a') as appender:
        appender.write(mtx1)
        appender.write(mtx2)
        appender.write(mtx3)
        appender.write(mtx4)
        appender.write(mtx5)
        appender.write(mtx6)
        appender.write(mtx7)
        appender.write(mtx8)
    
    return format(iou[1],'.4f'),format(loss_avg/val_loader.n_iter,'.4f')

def main():
    model_name_all = 'model.' + model_name
    # model = eval(model_name_all)(n_class=n_class)
  
    model = eval(model_name_all)(out_chans)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3,  512, 512), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    from thop import profile
    flops, params = profile(model, inputs=(torch.randn(1, 3, 512,512),))
    print('      - Flops: %0.2f ' %(flops/1e9))
    print('      - Params: %0.2f '%(params/1e6))
    # gpus = [1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    # model = nn.DataParallel(model.cuda(),device_ids=gpus,output_device=gpus[0])
    if gpu >= 0: model.cuda(gpu)

    optimizer = eval(opt_name)
    
    # if continue_checkpoint_model_file_name!='':
    #     model.load_state_dict(torch.load(
    #     continue_checkpoint_model_file_name, map_location='cuda:0'))
    # if continue_checkpoint_optim_file_name!='':
    #     optimizer.load_state_dict(torch.load(continue_checkpoint_optim_file_name))

    if epoch_from > 1:
        print('| loading checkpoint file %s... ' % continue_checkpoint_model_file_name, end='')
        model.load_state_dict(torch.load(continue_checkpoint_model_file_name, map_location='cuda:0'))
        # optimizer.load_state_dict(torch.load(continue_checkpoint_optim_file_name))
        print('done!')

    train_dataset = My_dataset(map_dir = train_map, map_seffix = map_seffix,label_dir = train_label, label_seffix = label_seffix, class_num = n_class, have_label = True, input_h= in_wh,input_w=in_wh ,transform=augmentation_methods)

    val_dataset = My_dataset(map_dir=val_map, map_seffix=map_seffix, label_dir=val_label, label_seffix=label_seffix, class_num=n_class, have_label=True, input_h=in_wh, input_w=in_wh, transform=[])

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)

    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    stop_flag = False
    current_vacc = 0
    for epo in tqdm(range(epoch_from, epoch_max+1)):
        print('\n| epo #%s begin...' % epo)

        t_iou,t_loss = train(epo, model, train_loader, optimizer)
        v_iou,v_loss = validation(epo, model, val_loader)
        
        # record the score to tensorboard
        writer.add_scalars('train_iou',{project_name:float(t_iou)},epo)
        writer.add_scalars('train_loss',{project_name:float(t_loss)},epo)
        writer.add_scalars('val_iou',{project_name:float(v_iou)},epo)
        writer.add_scalars('val_loss',{project_name:float(v_loss)},epo)

        torch.save(model.state_dict(), checkpoint_model_file)

        if float(v_iou) <= current_vacc:
            continue
        current_vacc = max(current_vacc,float(v_iou))
        
        print('| saving check point model file... ', end='')
        
        checkpoint_epoch_name = model_dir_path + model_name  +'_epo' + str(epo) + '_tiou' + str(t_iou) +'_viou' +str(v_iou) +'.pth'
        torch.save(model.state_dict(), checkpoint_epoch_name)
        
        # torch.save(optimizer.state_dict(), checkpoint_optim_file)
        # optimize_epoch_name = model_dir_path + model_name  +'_epo' + str(epo) + '_tacc' + str(t_acc) +'_vacc' +str(v_acc) +'.optim'
        # torch.save(optimizer.state_dict(), optimize_epoch_name)

        print('done!')
        if stop_flag == True:
            break        
    writer.close()
    os.rename(checkpoint_model_file, final_model_file)

if __name__ == '__main__':
    model_dir_path = os.path.join(model_dir,project_name+'/')
    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(tensorboard_log_dir,exist_ok=True)
    
    checkpoint_model_file = os.path.join(model_dir_path, 'tmp.pth')
    checkpoint_optim_file = os.path.join(model_dir_path, 'tmp.optim')
    
    final_model_file      = os.path.join(model_dir_path, 'final.pth')
    log_file              = os.path.join(model_dir_path, 'log.txt')

    print('| training %s on GPU #%d with pytorch' % (model_name, gpu))
    print('| from epoch %d / %s' % (epoch_from, epoch_max))
    print('| model will be saved in: %s' % model_dir_path)

    main()
