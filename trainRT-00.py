# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from ultralytics import YOLOMM,RTDETRMM
'''
if __name__ == '__main__':
    model = YOLOMM('/opt/data/private/dmt/MM4/ultralytics/cfg/models/mine/yolo11-mm-EVS2+EVS0+FG.yaml')
    # model = YOLOMM('yolo11n-mm-mid3.yaml')
    model.train(
         data='/opt/data/private/dmt/datasets/M3FD_yolo/M3FD_yolo.yaml',
        imgsz=640,
        # ====== 优化器与学习率 (解决震荡与Batch16掉点) ======
                # ================= 周期与节拍 =================
        epochs=300,             # 通用黄金周期，给复杂数据集充分时间
        batch=8,
        patience=30,            # 增加容错震荡空间
        close_mosaic=15,
        # ================= 优化器与防爆机制 (RT-DETR 铁律保持不变) =================
        #optimizer='AdamW',
        #lr0=0.0002,
        #lrf=0.01,
        cos_lr=True,
        weight_decay=0.0001,
        warmup_epochs=5.0,
        warmup_bias_lr=0.05,
        # ================= 损失函数 (多类别均衡配置) =================
        single_cls=False,       # 关闭单类模式，正常进行多分类
        box=7.5,                # 恢复正常的回归权重
        dfl=1.5,
        cls=0.5,
        # ================= 数据增强 (中强度万金油) =================
        hsv_h=0.015,            
        hsv_s=0.4,              # 中度色彩/亮度扰动，对抗自然光照突变
        hsv_v=0.4,
        translate=0.1,          # 10% 随机平移
        scale=0.5,              # 50% 随机缩放 (对 DroneVehicle 小目标极其重要)
        #fliplr=0.5,             # 50% 水平翻转
        flipud=0.0,             # 注意：如果是 DroneVehicle 纯俯视视角，建议把它改成 0.5！
        mosaic=0.2,             # 100% 开启高强度马赛克             
        erasing=0.2,             

        exist_ok=False,
        project='runs/yolo11-mm-EVS2+EVS0+FG',
        name='M3FD')
'''
if __name__ == '__main__':
    model = YOLOMM('/opt/data/private/dmt/MM4/ultralytics/cfg/models/mine/yolo11-mm-EVS2+EVS0+cat.yaml')
    # model = YOLOMM('yolo11n-mm-mid3.yaml')
    model.train(
        # ================= 基础配置 =================
        data='/opt/data/private/dmt/datasets/LLVIP/LLVIP.yaml',
        device=0,               # 显卡ID
        workers=8,              # DataLoader 线程数
        exist_ok=True,

        # ================= 周期与节拍 =================
        epochs=150,             # LLVIP 120轮速通
        batch=8,                # 保持 Batch 8
        patience=20,            # 20轮不涨点早停
        close_mosaic=15,        # 最后15轮关闭马赛克，迎接飙升

        # ================= 优化器与防爆机制 (RT-DETR 铁律) =================
        #optimizer='AdamW',
        #lr0=0.0002,             # 绝对安全学习率
        #lrf=0.01,
        cos_lr=True,            # 余弦退火
        weight_decay=0.0001,
        warmup_epochs=5.0,      # 延长预热
        warmup_bias_lr=0.05,

        # ================= 损失函数 (LLVIP 单类专属) =================
        single_cls=True,        # LLVIP 只有行人，开启单类模式
        box=10.0,               # 强化边框回归
        dfl=2.0,                # 强化模糊边界惩罚
        cls=0.5,

        # ================= 数据增强 (低光照/红外专属) =================
        hsv_h=0.01,             # 极低色彩扰动
        hsv_s=0.2,
        hsv_v=0.2,
        erasing=0.2,            # 降低随机擦除，保护小目标
        mosaic=0.5,             # 降低马赛克强度，保护上下文
        mixup=0.05 ,             # 5%的混叠，对抗行人密集重叠
        project='runs/baseline_cut30',
        name='LLVIP')