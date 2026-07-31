import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/opt/data/private/dmt/MM5_TRE')
from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('/opt/data/private/dmt/MM5_TRE/ultralytics/cfg/models/mine/yolov11-mm-DGFG_re-xn-rgbs.yaml')
    model.train(
        data='/opt/data/private/dmt/datasets/FLIR-align-3class/FLIR_data.yaml',
        device=0, workers=8, exist_ok=True,
        pretrained='/opt/data/private/dmt/MM5/yolo11s.pt',
        epochs=200, batch=16, patience=30, close_mosaic=15,
        cos_lr=True, weight_decay=0.0001, warmup_epochs=5.0, warmup_bias_lr=0.05,
        box=10.0, dfl=2.0, cls=0.5, single_cls=False,
        hsv_h=0.01, hsv_s=0.2, hsv_v=0.2, erasing=0.2, mosaic=0.5, mixup=0.05,
        mm_ir_shift_p=0.5, mm_ir_shift_max=30,
        project='/opt/data/private/dmt/MM5_TRE/runs', name='FLIR_FDDet',
    )

