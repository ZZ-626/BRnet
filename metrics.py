
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    predicted = torch.round(output)
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    tp = np.sum(predicted * target)
    fp = np.sum(predicted * (1 - target))
    fn = np.sum((1 - predicted) * target)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return dice



def recall_s(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))
    recall = true_positives / (actual_positives + smooth)

    return recall


def precision_s(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positives = np.sum(np.round(np.clip(output * target, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(output, 0, 1)))
    precision = true_positives / (predicted_positives + smooth)
    return precision


def accuracy_s(output, target):
    predicted = torch.round(torch.sigmoid(output))
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    accuracy = np.mean(predicted == target)
    return accuracy

def f1_score_s(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    predicted = torch.round(output)
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    tp = np.sum(predicted * target)
    fp = np.sum(predicted * (1 - target))
    fn = np.sum((1 - predicted) * target)
    f1 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    return f1


def specificity_s(output, target):
    smooth = 1e-5
    predicted = torch.round(torch.sigmoid(output))
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_negatives = np.sum(np.round(np.clip((1 - predicted) * (1 - target), 0, 1)))
    actual_negatives = np.sum(np.round(np.clip(1 - target, 0, 1)))
    specificity = true_negatives / (actual_negatives + smooth)
    return specificity


def sensitivity_s(output, target):
    smooth = 1e-5
    predicted = torch.round(torch.sigmoid(output))
    predicted = predicted.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    true_positives = np.sum(np.round(np.clip(predicted * target, 0, 1)))
    actual_positives = np.sum(np.round(np.clip(target, 0, 1)))
    sensitivity = true_positives / (actual_positives + smooth)

    return sensitivity

def auc_score(output, target):
    target = (target > 0).float()
    output_prob = torch.sigmoid(output)
    target_np = target.cpu().numpy()
    output_prob_np = output_prob.detach().cpu().numpy()
    auc = roc_auc_score(target_np.flatten(), output_prob_np.flatten())

    return auc

def mcc_score(output, target):
    target = (target > 0).float()
    output_prob = torch.sigmoid(output)
    predicted = (output_prob > 0.5).float()
    target_array = target.squeeze(1).view(-1).cpu().numpy()
    predicted = predicted.squeeze(1).view(-1).cpu().numpy()
    mcc = matthews_corrcoef(target_array, predicted)
    return mcc


import time
import torch


def compute_complexity(model, input_size, device='cuda'):
    """
    使用 thop 计算模型的 FLOPs (G) 和 Params (M)
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError("Please install thop using 'pip install thop'")

    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    return flops / 1e9, params / 1e6


def compute_latency_fps(model, input_size, device='cuda', iterations=100):
    """
    使用 cuda.synchronize() 精确计算推理延迟 (ms) 和 FPS
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(50):
            model(dummy_input)
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps_value = iterations / total_time
    return avg_latency_ms, fps_value
from skimage.morphology import skeletonize


def cl_score(v, s):
    return np.sum(v * s) / (np.sum(s) + 1e-5)
def cldice_score(output, target):
    """
    计算 clDice (Centerline Dice)
    :param output: 模型输出的 logits (Batch, Channels, Height, Width)
    :param target: 真实标签 0/1 (Batch, Channels, Height, Width)
    """
    # 将模型输出转换为概率并二值化
    output_prob = torch.sigmoid(output).data.cpu().numpy()
    pred = (output_prob > 0.5).astype(np.uint8)
    gt = target.data.cpu().numpy().astype(np.uint8)

    batch_size = pred.shape[0]
    num_classes = pred.shape[1]

    cldice_list = []
    for b in range(batch_size):
        for c in range(num_classes):
            v_p = pred[b, c]  # 预测体积 (Volume predicted)
            v_l = gt[b, c]  # 真实体积 (Volume label)

            # 边界情况：如果预测和真实都完全为空（全黑），则得分为 1.0
            if np.sum(v_p) == 0 and np.sum(v_l) == 0:
                cldice_list.append(1.0)
                continue

            # 提取骨架 (Skeleton)
            # skimage 的 skeletonize 要求输入是 bool 或 0/1 的二值图
            s_p = skeletonize(v_p > 0).astype(np.uint8)
            s_l = skeletonize(v_l > 0).astype(np.uint8)

            # 计算拓扑精确率 (Topology Precision) 和拓扑敏感度 (Topology Sensitivity)
            t_prec = cl_score(v_l, s_p)
            t_sens = cl_score(v_p, s_l)

            # 计算 clDice
            if t_prec + t_sens == 0:
                cldice = 0.0
            else:
                cldice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens)

            cldice_list.append(cldice)

    # 返回整个 batch 的平均 clDice
    if len(cldice_list) == 0:
        return 0.0
    return np.mean(cldice_list)