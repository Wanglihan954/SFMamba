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
# 导入你的 JamMa 模型和配置
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from utils import EffectiveReceiptiveField # 原作者的归一化工具

# ========================================================
# 1. 核心包装器：瞒天过海，实现双图拼接求导
# ========================================================
class JamMaConcatERFWrapper(nn.Module):
    def __init__(self, pl_model):
        super().__init__()
        self.backbone = pl_model.backbone
        self.matcher = pl_model.matcher
        self.mp = pl_model.config.JAMMA.MP

    def forward(self, samples):
        """
        samples: [B, 3, H, 2W] (例如 512x1024 的拼接长图)
        """
        B, C, H, W2 = samples.shape
        W = W2 // 2

        # 切分长图为左图和右图
        batch = {
            'imagec_0': samples[:, :, :, :W],
            'imagec_1': samples[:, :, :, W:],
        }

        with torch.autocast(enabled=self.mp, device_type='cuda'):
            # 跑 Backbone
            self.backbone(batch)
            
            # 补齐 Mamba 需要的尺寸参数
            batch.update({
                'hw0_i': batch['imagec_0'].shape[2:],
                'hw1_i': batch['imagec_1'].shape[2:],
                'hw0_c': [batch['h_8'], batch['w_8']],
                'hw1_c': [batch['h_8'], batch['w_8']],
            })
            
            # 只跑核心的粗匹配（包含 JointMamba 的全局全向扫描 JEGO）
            self.matcher.coarse_match(batch)

        # 提取经过 Mamba 交互后的 1D 序列特征
        feat0_seq = batch['feat_8_0'] 
        feat1_seq = batch['feat_8_1']
        
        # 折叠回 2D 空间特征图 [B, Channels, H/8, W/8]
        h_8, w_8 = batch['h_8'], batch['w_8']
        feat0_2d = feat0_seq.view(B, -1, h_8, w_8)
        feat1_2d = feat1_seq.view(B, -1, h_8, w_8)
        
        # 横向拼接为长宽比 1:2 的完整特征图
        concat_feat = torch.cat([feat0_2d, feat1_2d], dim=3)
        
        return concat_feat

# ========================================================
# 2. 感受野求导与计算核心逻辑
# ========================================================
def get_custom_point_grad(model, samples, h_ratio, w_ratio):
    outputs = model(samples)
    out_size = outputs.size()
    target_h = int((out_size[2] - 1) * h_ratio)
    target_w = int((out_size[3] - 1) * w_ratio)
    
    # 锁定目标像素点求导
    target_point = torch.nn.functional.relu(outputs[:, :, target_h, target_w]).sum()
    grad = torch.autograd.grad(target_point, samples)[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    return aggregated.cpu().numpy()

def compute_erf_for_point(model, data_path, image_height, image_width, h_ratio, w_ratio, num_images=50):
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    val_dir = os.path.join(data_path, 'val') if os.path.exists(os.path.join(data_path, 'val')) else data_path
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=1, pin_memory=True)

    meter = AverageMeter()
    model.cuda().eval()
    
    print(f"正在计算点位 (h:{h_ratio:.2f}, w:{w_ratio:.2f}) 的 ERF (跑 {num_images} 张图取平均)...")
    for _, (samples, _) in tqdm.tqdm(enumerate(data_loader), total=num_images):
        if meter.count >= num_images: break
        samples = samples.cuda(non_blocking=True).requires_grad_()
        contribution_scores = get_custom_point_grad(model, samples, h_ratio, w_ratio)
        if not np.isnan(np.sum(contribution_scores)):
            meter.update(contribution_scores)
            
    return EffectiveReceiptiveField.simpnorm(meter.avg)

# ========================================================
# 3. 完美复现论文的“原尺寸无边框”画图函数
# ========================================================
def plot_erf_with_red_circle_exact(erf_matrix, h_ratio, w_ratio, save_path):
    H, W = erf_matrix.shape
    circle_y = int((H - 1) * h_ratio)
    circle_x = int((W - 1) * w_ratio)

    dpi = 100 
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off') 
    
    # 画绿色热力图
    ax.imshow(erf_matrix, cmap='YlGn', aspect='auto', vmin=0.0, vmax=1.0)
    
    # 画红色空心圆圈（查询点）
    ax.plot(circle_x, circle_y, marker='o', color='none', markeredgecolor='red', 
            markersize=12, markeredgewidth=2.0)
    
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"-> 成功保存图片: {save_path}\n")

# ========================================================
# 4. 主函数：解析参数并执行
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_cfg_path', type=str,default="configs/data/megadepth_test_1500.py", help='data config path')
    parser.add_argument('--main_cfg_path', type=str,default="configs/jamma/outdoor/test.py", help='main config path')
    parser.add_argument('--ckpt_path', type=str, default="/home/ly/9100pro/Reg/JamMa/jamma.ckpt", help='model weights')
    parser.add_argument('--erf_data_path', type=str, default='/home/ly/9100pro/Reg/JamMa/ERF/test_images', help='图片文件夹路径 (用于计算平均梯度)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 初始化配置
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    # 加载你的 JamMa 模型
    print("正在加载 JamMa 模型...")
    pl_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path)
    
    # 挂载我们写好的 ERF Wrapper
    model = JamMaConcatERFWrapper(pl_model).cuda().eval()
    
    # 设定长宽比例 (1:2 拼接图)
    IMAGE_H, IMAGE_W = 512, 1024 
    save_dir = "./jamma_erf_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 完美对齐你论文图里的 3 个点位
    # 注意：整张拼接长图的宽度比例 w_ratio 是 0.0 到 1.0。
    # 其中 0.0~0.5 属于左图，0.5~1.0 属于右图。
    points = [
        # Point 1: 左图的左上角 (对应图1第一行)
        {"name": "Point_1_TopLeft", "h_ratio": 0.05, "w_ratio": 0.025}, 
        
        # Point 2: 左图的右下角 (对应图1第二行)
        {"name": "Point_2_BottomRight", "h_ratio": 0.95, "w_ratio": 0.475}, 
        
        # Point 3: 左图的正中心 (对应图1第三行)
        {"name": "Point_3_Center", "h_ratio": 0.50, "w_ratio": 0.250}, 
    ]
    
    print("\n================ 开始计算 ERF ================")
    for pt in points:
        erf_matrix = compute_erf_for_point(
            model=model, 
            data_path=args.erf_data_path, 
            image_height=IMAGE_H, 
            image_width=IMAGE_W,
            h_ratio=pt["h_ratio"], 
            w_ratio=pt["w_ratio"],
            num_images=50  # 为了图好看平滑，跑 50 张图取平均
        )
        
        save_path = os.path.join(save_dir, f"JamMa_{pt['name']}.jpg")
        plot_erf_with_red_circle_exact(erf_matrix, pt["h_ratio"], pt["w_ratio"], save_path)
        
    print("================ 全部完成！ ================")
    print(f"请前往 {save_dir} 文件夹查看结果。")