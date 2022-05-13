# create by andy at 2022/5/13
# reference:
from typing import Optional

from torch.nn import Conv2d
from torchvision.models import resnet
from torchvision.models.segmentation.fcn import _fcn_resnet, FCN


def my_fcn_resnet50(
        num_classes: int = 5,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> FCN:
    backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    backbone.conv1 = Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    return model




if __name__ == '__main__':
    a = my_fcn_resnet50()
    print(a.name)
