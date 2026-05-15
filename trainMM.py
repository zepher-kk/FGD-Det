#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11n-mm-mid.yaml')
    # model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ultralytics/cfg/models/mm/yolo11n-mm-mid.yaml')
    model.train(data=r"D:\learn\dmt\datasets\M3FD_yolo_10percent\data.yaml",
                epochs=50,batch=2,
                scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                # modality='RGB', # 模态消融参数 非必要不得开启（rgb/RGB、x/X token 大小写不敏感，内部统一显示为 RGB/X）
                # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
                # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
                # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
                amp=True,  #启动amp混合精度加速
                device='0',
                cache=True,
                workers = 2,
                exist_ok=True,
                # afss_enabled=True, #启动afss训练采样机制 
                project='ResTest',name='AFSStest-non')
