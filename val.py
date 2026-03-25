import argparse
import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm
from metrics import iou_score, dice_coef, recall_s, precision_s, accuracy_s,  f1_score_s, specificity_s, \
    sensitivity_s, auc_score, mcc_score, compute_complexity,  compute_latency_fps
from metrics import cldice_score
import archs
from dataset import Dataset
from utils import AverageMeter
import albumentations as albu

"""                 
需要指定参数：--name arcade_BiFormer_resnet18_woDSBEST
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="arcade_BiFormer_resnet18_woDSBEST",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'UNet':
        model = archs.__dict__[config['arch']](config['num_classes'])
    elif config['arch'] == 'BiFormer_resnet18':
        model = archs.__dict__[config['arch']]()
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision']
                                               )

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('arcade', "test", 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = img_ids

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('arcade', "test", 'images'),
        mask_dir=os.path.join('arcade', "test", 'masks'),

        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],

        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    avg_dice_meter = AverageMeter()

    avg_recall_meter = AverageMeter()
    avg_precision_meter = AverageMeter()
    avg_accuracy_meter = AverageMeter()
    avg_f1_score_meter = AverageMeter()
    avg_specificity_meter = AverageMeter()
    avg_sensitivity_meter = AverageMeter()
    avg_auc_meter = AverageMeter()
    avg_mcc_meter = AverageMeter()

    # 【新增这一行】
    avg_cldice_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    with torch.no_grad():

        # ==========================================================
        # 在验证循环【外】集中计算 Params, FLOPs, Latency 和 FPS
        # ==========================================================
        print("=> Calculating Model Complexity & Speed...")
        # 假设 batch_size=1 时测速最能反映真实推理延迟
        dummy_input_size = (1, config['input_channels'], config['input_h'], config['input_w'])

        # 1. 计算 FLOPs 和 Params
        flops_g, params_m = compute_complexity(model, dummy_input_size, device='cuda')

        # 2. 计算 延迟和 FPS
        latency_ms, fps_value = compute_latency_fps(model, dummy_input_size, device='cuda')

        print(f"Model FLOPs: {flops_g:.2f} G")
        print(f"Model Params: {params_m:.2f} M")
        print(f"Inference Latency: {latency_ms:.2f} ms")
        print(f"Model FPS: {fps_value:.2f}")
        # ==========================================================

        # # 计算模型参数数量
        # params_value = model_params(model)
        # avg_params_meter.update(params_value, 1)  # 固定值，更新一次即可

        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            dice = dice_coef(output, target)
            avg_dice_meter.update(dice, input.size(0))

            recall = recall_s(output, target)
            avg_recall_meter.update(recall, input.size(0))

            precision = precision_s(output, target)
            avg_precision_meter.update(precision, input.size(0))

            accuracy = accuracy_s(output, target)
            avg_accuracy_meter.update(accuracy, input.size(0))

            f1_score = f1_score_s(output, target)
            avg_f1_score_meter.update(f1_score, input.size(0))
            specificity = specificity_s(output, target)
            avg_specificity_meter.update(specificity, input.size(0))
            sensitivity = sensitivity_s(output, target)
            avg_sensitivity_meter.update(sensitivity, input.size(0))

            auc = auc_score(output, target)
            avg_auc_meter.update(auc, input.size(0))
            mcc = mcc_score(output, target)
            avg_mcc_meter.update(mcc, input.size(0))


            # 【新增这两行，计算 clDice】
            cldice = cldice_score(output, target)
            avg_cldice_meter.update(cldice, input.size(0))


            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    print('Dice: %.4f' % avg_dice_meter.avg)
    print('recall: %.4f' % avg_recall_meter.avg)
    print('precision: %.4f' % avg_precision_meter.avg)
    print('accuracy: %.4f' % avg_accuracy_meter.avg)

    print('sensitivity: %.4f' % avg_sensitivity_meter.avg)
    print('specificity: %.4f' % avg_specificity_meter.avg)
    print('auc: %.4f' % avg_auc_meter.avg)
    print('mcc: %.4f' % avg_mcc_meter.avg)

    print('f1_score: %.4f' % avg_f1_score_meter.avg)
    # 【新增这一行】
    print('clDice: %.4f' % avg_cldice_meter.avg)
    # 保存差异图到output2文件夹
    save_difference_maps(val_loader, model, config)
    torch.cuda.empty_cache()


def save_difference_maps(val_loader, model, config):
    """保存分割结果与真实值之间的差异图到output2文件夹"""

    # 创建输出文件夹
    output_dir = os.path.join('output2', config['name'])
    os.makedirs(output_dir, exist_ok=True)

    # 为每种类别创建子文件夹
    for c in range(config['num_classes']):
        os.makedirs(os.path.join(output_dir, str(c)), exist_ok=True)

    model.eval()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader), desc="Saving difference maps"):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            # 获取预测结果
            pred = torch.sigmoid(output).cpu().numpy()
            target_np = target.cpu().numpy()

            for i in range(len(pred)):
                img_id = meta['img_id'][i]

                for c in range(config['num_classes']):
                    # 获取当前类别的预测和真实标签
                    pred_binary = (pred[i, c] > 0.40).astype(int)
                    target_binary = target_np[i, c].astype(int)

                    # 创建差异图 (BGR格式)
                    difference_map = np.zeros((target_binary.shape[0], target_binary.shape[1], 3), dtype=np.uint8)

                    # 真阳性 (TP): 预测为1，实际为1 - 绿色
                    tp = (pred_binary == 1) & (target_binary == 1)
                    difference_map[tp] = [0, 255, 0]  # 绿色

                    # 假阴性 (FN): 预测为0，实际为1 - 蓝色
                    fn = (pred_binary == 0) & (target_binary == 1)
                    difference_map[fn] = [255, 0, 0]  # 蓝色

                    # 假阳性 (FP): 预测为1，实际为0 - 红色
                    fp = (pred_binary == 1) & (target_binary == 0)
                    difference_map[fp] = [0, 0, 255]  # 红色

                    # 保存差异图
                    diff_path = os.path.join(output_dir, str(c), img_id + '_diff.jpg')
                    cv2.imwrite(diff_path, difference_map)

                    # 同时保存原始预测图和叠加图
                    overlay_path = os.path.join(output_dir, str(c), img_id + '_overlay.jpg')
                    overlay = create_overlay_image(pred_binary, target_binary)
                    cv2.imwrite(overlay_path, overlay)


def create_overlay_image(pred_binary, target_binary):
    """创建预测和真实的叠加图"""
    overlay = np.zeros((target_binary.shape[0], target_binary.shape[1], 3), dtype=np.uint8)

    # 真实标签用白色显示
    overlay[target_binary == 1] = [255, 255, 255]

    # 预测结果用半透明红色叠加
    pred_area = (pred_binary == 1)
    overlay[pred_area] = overlay[pred_area] * 0.5 + np.array([0, 0, 255]) * 0.5

    return overlay.astype(np.uint8)

if __name__ == '__main__':
    main()
