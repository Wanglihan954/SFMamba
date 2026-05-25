import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from utils import EffectiveReceiptiveField 

# ========================================================
# 1. 纯净包装器：原汁原味输入两张独立图，输出左图特征
# ========================================================
class JamMaIndependentERFWrapper(nn.Module):
    def __init__(self, pl_model):
        super().__init__()
        self.backbone = pl_model.backbone
        self.matcher = pl_model.matcher
        self.mp = pl_model.config.JAMMA.MP

    def forward(self, imgA, imgB):
        # 100% 独立的输入字典
        batch = {
            'imagec_0': imgA,
            'imagec_1': imgB,
        }

        with torch.autocast(enabled=self.mp, device_type='cuda'):
            self.backbone(batch)
            batch.update({
                'hw0_i': batch['imagec_0'].shape[2:],
                'hw1_i': batch['imagec_1'].shape[2:],
                'hw0_c': [batch['h_8'], batch['w_8']],
                'hw1_c': [batch['h_8'], batch['w_8']],
            })
            self.matcher.coarse_match(batch)

        # 提取左图（图 A）经过 Mamba 全局交互后的特征
        feat0_seq = batch['feat_8_0'] 
        B = imgA.shape[0]
        h_8, w_8 = batch['h_8'], batch['w_8']
        featA_2d = feat0_seq.view(B, -1, h_8, w_8)
        
        return featA_2d

# ========================================================
# 2. 【核心魔法】独立双重求导：一个点，同时对两张图求导
# ========================================================
def get_dual_point_grad(model, imgA, imgB, h_ratio, w_ratio):
    # 前向传播，得到左图的特征
    featA = model(imgA, imgB)
    
    # 锁定左图上的目标点
    out_size = featA.size()
    target_h = int((out_size[2] - 1) * h_ratio)
    target_w = int((out_size[3] - 1) * w_ratio)
    target_point = torch.nn.functional.relu(featA[:, :, target_h, target_w]).sum()
    
    # ----------------------------------------------------
    # 高能：向 PyTorch 申请同时对 imgA 和 imgB 算梯度！
    # ----------------------------------------------------
    grads = torch.autograd.grad(target_point, (imgA, imgB))
    gradA = grads[0] # 左图对这个点的影响（自身感受野）
    gradB = grads[1] # 右图对这个点的影响（跨图感受野）
    
    # 分别清理负值并压缩通道
    gradA = torch.nn.functional.relu(gradA).sum((0, 1))
    gradB = torch.nn.functional.relu(gradB).sum((0, 1))
    
    # 为了画图好看，把这两个独立的 512x512 矩阵，横向拼成一个 512x1024 的矩阵
    combined_grad = torch.cat([gradA, gradB], dim=1)
    
    return combined_grad.cpu().numpy()

# ========================================================
# 3. 计算流：每次抽取两张独立的图进行测试
# ========================================================
def compute_dual_erf(model, data_path, image_size, h_ratio, w_ratio, num_images=50):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    val_dir = os.path.join(data_path, 'val') if os.path.exists(os.path.join(data_path, 'val')) else data_path
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    # 每次抽 2 张不同的图
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=2, drop_last=True)

    meter = AverageMeter()
    model.cuda().eval()
    
    print(f"正在计算图A点(h:{h_ratio:.2f}, w:{w_ratio:.2f}) 的双重 ERF...")
    for _, (samples, _) in tqdm.tqdm(enumerate(data_loader), total=num_images):
        if meter.count >= num_images: break
        
        # 100% 独立的图 A 和 图 B
        imgA = samples[0:1].cuda(non_blocking=True)
        imgB = samples[1:2].cuda(non_blocking=True)
        
        # 两张图都需要挂载梯度！
        imgA.requires_grad_()
        imgB.requires_grad_()
        
        # 送入双重求导器
        contribution_scores = get_dual_point_grad(model, imgA, imgB, h_ratio, w_ratio)
        if not np.isnan(np.sum(contribution_scores)):
            meter.update(contribution_scores)
            
    # 执行 0.2 次方平滑归一化
    return EffectiveReceiptiveField.simpnorm(meter.avg)

# ========================================================
# 4. 画图函数：打上红圈
# ========================================================
def plot_dual_erf_with_circle(erf_matrix, h_ratio, w_ratio, save_path):
    H, W_total = erf_matrix.shape
    W_single = W_total // 2 # 单张图的宽度
    
    # 计算红圈在左图上的绝对坐标
    circle_y = int((H - 1) * h_ratio)
    circle_x = int((W_single - 1) * w_ratio)

    dpi = 100 
    fig = plt.figure(figsize=(W_total / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off') 
    
    ax.imshow(erf_matrix, cmap='YlGn', aspect='auto', vmin=0.0, vmax=1.0)
    
    # 在左半边画上红圈
    ax.plot(circle_x, circle_y, marker='o', color='none', markeredgecolor='red', 
            markersize=12, markeredgewidth=2.0)
    
    plt.savefig(save_path, dpi=dpi)
    plt.close()

# ========================================================
# 5. 主函数
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument('--ckpt_path', type=str, default="/home/ly/9100pro/Reg/JamMa/jamma.ckpt")
    parser.add_argument('--erf_data_path', type=str, default='/home/ly/9100pro/Reg/JamMa/ERF/test_images', help='图片文件夹路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    pl_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path)
    model = JamMaIndependentERFWrapper(pl_model).cuda().eval()
    
    IMAGE_SIZE = 512 # 独立正方形输入
    save_dir = "./jamma_dual_erf"
    os.makedirs(save_dir, exist_ok=True)
    
    # 目标点在图 A (左图) 的相对比例
    points = [
        {"name": "Point_1_TopLeft", "h_ratio": 0.05, "w_ratio": 0.05}, 
        {"name": "Point_2_BottomRight", "h_ratio": 0.95, "w_ratio": 0.95}, 
        {"name": "Point_3_Center", "h_ratio": 0.50, "w_ratio": 0.50}, 
    ]
    
    for pt in points:
        # 拿到横向拼接好的梯度矩阵 (512 x 1024)
        erf_matrix = compute_dual_erf(
            model=model, 
            data_path=args.erf_data_path, 
            image_size=IMAGE_SIZE, 
            h_ratio=pt["h_ratio"], 
            w_ratio=pt["w_ratio"],
            num_images=50  
        )
        
        save_path = os.path.join(save_dir, f"JamMa_DualERF_{pt['name']}.jpg")
        plot_dual_erf_with_circle(erf_matrix, pt["h_ratio"], pt["w_ratio"], save_path)
        print(f"-> 成功保存双重感受野: {save_path}\n")