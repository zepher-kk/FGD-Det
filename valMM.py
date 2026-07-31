from ultralytics import YOLOMM

model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM-LST/weights/best.pt')
model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
          split='test',device='0',
        #   modality='x',模态消融参数 非必要不得开启
          # coco=True,
          project='ResTest',
          name='test-mm-asymmetric-val')
