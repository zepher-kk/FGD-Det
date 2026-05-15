# FSCF 风格多模态配置

本目录存放 `FSConv`、`ACA`、`MRCB` 在 `YOLO11mm` 与 `YOLOMM-10` 上的配置扩展。

约定如下：

- `yolo11n-mm-fscf.yaml`：`YOLO11mm` 组合版，联合启用 `FSConv`、`ACA`、`MRCB`。
- `yolo11n-mm-fsconv.yaml`：`YOLO11mm` 仅 `FSConv` 版本。
- `yolo11n-mm-aca.yaml`：`YOLO11mm` 仅 `ACA` 版本。
- `yolo11n-mm-mrcb.yaml`：`YOLO11mm` 仅 `MRCB` 版本。

这些 `YOLO11mm` 配置用于结构级验证，**不混入 `loss_box: nwd`**，避免结构改动与损失改动相互干扰。

最终 `YOLOMM-10` 论文风格版本位于：

- `ultralytics/cfg/models/mm/v10/yolov10n-mm-fscf.yaml`

该最终版本在训练时应显式传入 `loss_box=nwd`，而不是修改全局默认配置。
