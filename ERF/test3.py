import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt
os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ========================================================
# 导入模型与【原作者的工具类】
# ========================================================
from src.config.default import get_cfg_defaults
from src.lightning.lightning_jamma import PL_JamMa
from utils import EffectiveReceiptiveField, visualize 

# ========================================================
# 1. 长图拼接 Wrapper (为了配合原作者单输入求导机制)
# ========================================================
class JamMaConcatERFWrapper(nn.Module):
    def __init__(self, pl_model):
        super().__init__()
        self.backbone = pl_model.backbone
        self.matcher = pl_model.matcher
        self.mp = pl_model.config.JAMMA.MP

    def forward(self, samples):
        B, C, H, W2 = samples.shape
        W = W2 // 2

        batch = {
            'imagec_0': samples[:, :, :, :W],
            'imagec_1': samples[:, :, :, W:],
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

        feat0_seq = batch['feat_8_0'] 
        feat1_seq = batch['feat_8_1']
        
        h_8, w_8 = batch['h_8'], batch['w_8']
        feat0_2d = feat0_seq.view(B, -1, h_8, w_8)
        feat1_2d = feat1_seq.view(B, -1, h_8, w_8)
        
        concat_feat = torch.cat([feat0_2d, feat1_2d], dim=3)
        return concat_feat

# ========================================================
# 2. 自定义点位求导逻辑
# ========================================================
def get_custom_point_grad(model, samples, h_ratio, w_ratio):
    outputs = model(samples)
    out_size = outputs.size()
    # 根据比例计算目标点的物理坐标
    target_h = int((out_size[2] - 1) * h_ratio)
    target_w = int((out_size[3] - 1) * w_ratio)
    
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
    
    print(f"\n正在计算点位 (h:{h_ratio:.2f}, w:{w_ratio:.2f}) 的感受野梯度...")
    for _, (samples, _) in tqdm.tqdm(enumerate(data_loader), total=num_images):
        if meter.count >= num_images: break
        samples = samples.cuda(non_blocking=True).requires_grad_()
        contribution_scores = get_custom_point_grad(model, samples, h_ratio, w_ratio)
        if not np.isnan(np.sum(contribution_scores)):
            meter.update(contribution_scores)
            
    # 【调用原作者的 5 次方根平滑归一化】
    return EffectiveReceiptiveField.simpnorm(meter.avg)

# ========================================================
# 3. 【核心】使用原作者 visualize 工具进行复刻画图
# ========================================================
def plot_author_style_erf(erf_results, save_path):
    """
    使用原作者的 seaborn_heatmap 和 RdYlGn 配色
    erf_results: 列表，包含字典 [{'matrix': data, 'h_ratio': h, 'w_ratio': w, 'title': title}]
    """
    rows = len(erf_results)
    # 计算画布大小 (保持原图比例)
    figsize = (10, 3 * rows)
    fig, axs = plt.subplots(rows, 1, squeeze=False, figsize=figsize, dpi=300)
    
    for i, res in enumerate(erf_results):
        matrix = res['matrix']
        h_ratio = res['h_ratio']
        w_ratio = res['w_ratio']
        H, W = matrix.shape
        
        # 提取像素坐标
        circle_y = int((H - 1) * h_ratio)
        circle_x = int((W - 1) * w_ratio)
        
        ax = axs[i, 0]
        
        # 【完全复刻原作者调用】：利用 seanborn_heatmap 画底图
        # 原作者的 center=0 配合 RdYlGn，会让 0 变成黄色，1 变成绿色
        _, mesh = visualize.seanborn_heatmap(
            matrix, xticklabels=False, yticklabels=False, 
            vmin=0.0, vmax=1.0, cmap='RdYlGn', 
            center=0, annot=False, ax=ax, 
            cbar=False, annot_kws={"size": 24}, fmt='.2f'
        )
        
        # 叠加论文中的红色空心圆圈
        ax.plot(circle_x, circle_y, marker='o', color='none', markeredgecolor='red', 
                markersize=15, markeredgewidth=2.5)
        
        # 去掉边框和坐标轴，保持论文纯净感
        ax.set_axis_off()
        ax.set_title(res['title'], fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\n-> 原汁原味的热力图已保存至: {save_path}")

# ========================================================
# 4. 主函数
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_cfg_path', type=str, default="configs/data/megadepth_test_1500.py")
    parser.add_argument('--main_cfg_path', type=str, default="configs/jamma/outdoor/test.py")
    parser.add_argument('--ckpt_path', type=str, default="/home/ly/9100pro/Reg/JamMa/jamma.ckpt")
    parser.add_argument('--erf_data_path', type=str, default='/home/ly/9100pro/Reg/JamMa/ERF/test_images' , help='图片文件夹路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 初始化模型
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    print("正在加载 JamMa 模型...")
    pl_model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path)
    model = JamMaConcatERFWrapper(pl_model).cuda().eval()
    
    # 定义长图尺寸 (1:2)
    IMAGE_H, IMAGE_W = 512, 1024 
    
    # 【精确对齐论文原图的点位】
    points = [
        # 行1：左图极左上角 -> 对应长图的 (0.0, 0.0)
        {"title": "Point 1: Top-Left", "h_ratio": 0.0, "w_ratio": 0.0}, 
        
        # 行2：右图极右下角 -> 对应长图的 (1.0, 1.0)
        {"title": "Point 2: Bottom-Right", "h_ratio": 1.0, "w_ratio": 1.0}, 
        
        # 行3：左图正中心 -> 对应长图高度的 0.5，宽度的 0.25 (1/4处)
        {"title": "Point 3: Center-Left", "h_ratio": 0.50, "w_ratio": 0.25}, 
    ]
    
    erf_results = []
    print("\n================ 开始计算 ERF ================")
    for pt in points:
        matrix = compute_erf_for_point(
            model=model, 
            data_path=args.erf_data_path, 
            image_height=IMAGE_H, 
            image_width=IMAGE_W,
            h_ratio=pt["h_ratio"], 
            w_ratio=pt["w_ratio"],
            num_images=50  
        )
        erf_results.append({
            'matrix': matrix,
            'h_ratio': pt["h_ratio"],
            'w_ratio': pt["w_ratio"],
            'title': pt["title"]
        })
        
    save_dir = "./jamma_paper_erf"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "JamMa_Paper_Style2.png")
    
    # 执行画图
    plot_author_style_erf(erf_results, save_path)
    print("================ 全部完成！ ================")