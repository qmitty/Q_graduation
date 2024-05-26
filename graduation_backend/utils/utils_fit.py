import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr, cvtColor, preprocess_input, resize_image, show_config
from utils.utils_metrics import f_score, mcc_score, dice_score

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ----------------------------------已改为多模态输入------------------------------ #
def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0  

    file_name = "result.csv"
    # 使用os.path.join组合完整的文件路径
    csv_file = os.path.join(save_dir, file_name)
    

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs_A, imgs_B, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs_A   = imgs_A.cuda(local_rank)
                imgs_B   = imgs_B.cuda(local_rank)
                pngs     = pngs.cuda(local_rank)
                labels   = labels.cuda(local_rank)
                weights  = weights.cuda(local_rank)
                # image    = cvtColor(imgs_A)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs_A, imgs_B)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score、MCC、dice_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)
                _mcc = mcc_score(outputs, labels)
                _dice = dice_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs_A, imgs_B)

                #----------------------#
                #   损失计算
                #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score、MCC、dice
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)
                    _mcc = mcc_score(outputs, labels)
                    _dice = dice_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        # 在每个epoch结束时计算并记录loss和f_score到csv表中
        loss_value = total_loss / epoch_step
        f_score_value = total_f_score / epoch_step
        epoch_data = {'Epoch': epoch + 1, 'Loss': loss_value, 'F-Score': f_score_value, 'MCC': _mcc.item(), 'Dice': _dice.item()}
        epoch_df = pd.DataFrame([epoch_data])  # 直接创建包含当前epoch数据的DataFrame

        # 检查文件是否存在，如果不存在，则先创建文件并写入表头
        if not os.path.isfile(csv_file):
            epoch_df.to_csv(csv_file, mode='w', header=True, index=False)
        else:
            epoch_df.to_csv(csv_file, mode='a', header=False, index=False)

        #----------------------#
        #   可视化
        #----------------------#

        pr = outputs[0]  
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().detach().numpy()
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        # 用于与原图混合的颜色设置
        colors = [ (0, 0, 0), (255, 140, 26), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        
        # 用于单独分割图的颜色设置
        # colors = [(0, 0, 0), (255, 255, 255)]  # 黑色和白色
        
        original_h, original_w = imgs_A.size(2), imgs_A.size(3)
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        seg_image_pil = Image.fromarray(np.uint8(seg_img))

        #------------------------------------------------#
        #   将Tensor转换为PIL图像以便进行混合
        #------------------------------------------------#
        imgs_A_pil = transforms.ToPILImage()(imgs_A[0].cpu())  # 假设我们只可视化第一个图像
        imgs_A_pil = imgs_A_pil.convert("RGB")  # 确保图像是RGB格式
        
        #------------------------------------------------#
        #   将新图与原图进行混合 或 只显示分割图
        #------------------------------------------------#
        blend_image = Image.blend(imgs_A_pil, seg_image_pil, 0.9)
        # blend_image   = Image.fromarray(np.uint8(seg_image_pil))

        # 后台显示混合后的图像
        # blend_image.show()
        # 保存图片
        save_path = os.path.join(save_dir, "segimage", f"visualization_{epoch + 1}.png")
        seg_image_pil.save(save_path)
        
        

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))