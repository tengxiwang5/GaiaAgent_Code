import torch
from torch import nn
from torch.nn import functional as F
from utils.misc import initialize_weights
from mobile import mobile_sam
from models.ContrastiveLearning import ContrastiveLearning
from models.InformationInteraction import InformationInteraction

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high 
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        # x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int, alpha: float = 1.0):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha

        # 冻结原始参数
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # 添加 LoRA 参数
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, original_layer.out_features) * 0.01)

        # 缩放系数
        self.scaling = alpha / r

    def forward(self, x):
        return self.original_layer(x) + self.scaling * (x @ self.lora_A @ self.lora_B)

def replace_with_lora(model, target_modules, r=16, alpha=32):
    """
    替换模型中的目标层为 LoRA 层
    Args:
        model (nn.Module): 原始模型
        target_modules (list[str]): 要替换的层名称
        r (int): LoRA 的秩
        alpha (float): 缩放系数
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            # 替换为 LoRA 版本
            setattr(model, name, LoRALinear(module, r, alpha))
        else:
            # 递归替换子模块
            replace_with_lora(module, target_modules, r, alpha)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # 全局平均池化，得到 [b, c]
        y = F.relu(self.fc1(y))           # 第一个全连接层
        y = torch.sigmoid(self.fc2(y))    # 第二个全连接层 + Sigmoid 激活
        y = y.view(b, c, 1, 1)            # 调整为 [b, c, 1, 1] 维度
        return x * y                      # 加权原特征图

class GaiaAgent_VLMTool(nn.Module):
    def __init__(
        self,
        num_embed=8,
        # model_name: str='FastSAM-x.pt',
        conf: float=0.4,
        iou: float=0.9,
        imgsz: int=1024,
        retina_masks: bool=True,
        done_warmup: bool=True,
        ):
        super(GaiaAgent_VLMTool, self).__init__()
        # self.model = FastSAM(model_name)
        self.sam = mobile_sam
        target_modules = ["qkv", "proj", "fc1", "fc2"]  # 根据实际模块名称修改
        # target_modules = ["fc1", "fc2"]  # 根据实际模块名称修改
        replace_with_lora(self.sam.image_encoder, target_modules, r=16, alpha=32)
        # 冻结非 LoRA 参数
        for name, param in self.sam.image_encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        

        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None 
        self.channelconv = nn.Sequential(
                                        nn.Conv2d(256, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(160),
                                        nn.ReLU(),
                                        SEBlock(160)
                                    )

        self.regressor = TwoLayerConv2d(in_channels=8, out_channels=1)
        
        self.active3d = nn.Tanh()
                                       
        self.Dec2 = _DecoderBlock(160, 64)
        self.information_interaction = InformationInteraction(64,4)
        self.resCD = self._make_layer(ResBlock, 64, 64, 6, stride=1)
        self.headC = nn.Sequential(nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.segmenterC = nn.Conv2d(8, 7, kernel_size=1)
        self.contrastive_learning = ContrastiveLearning(temperature=0.1)
                      
        initialize_weights(self.channelconv, self.Dec2, self.resCD, self.headC, self.segmenterC)

    def apply_sam(self, feature_map):
        # 使用SAM进行处理
        sam_features = self.sam.image_encoder(feature_map)
        return sam_features
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, text=None, train_mode=True):
        # print(f"x1: {x1.shape}")
    
        input_shape = x1.shape[-2:]
        # print(f"x1: {x1.shape}")
        featsA = self.apply_sam(x1)
        # print(f"featsA: {featsA.shape}")
        featsB = self.apply_sam(x2)

        featA_s16 = self.channelconv(featsA)
        
        decA_2 = self.Dec2(featA_s16)#160-80
        B, C, H, W = decA_2.shape
        decA_2_1  = decA_2.view(B, C, H * W).permute(0, 2, 1)  # (B, 4096, 256)
        # print(f"decA_2_1: {decA_2_1.shape}")
        decA_2 = decA_2_1.permute(0, 2, 1).contiguous().view(B, C, H, W)
        # print(f"decA_2: {decA_2.shape}")

        featB_s16 = self.channelconv(featsB.clone())
        
        decB_2 = self.Dec2(featB_s16)
        decB_2_1  = decB_2.view(B, C, H * W).permute(0, 2, 1)
        decB_2 = decB_2_1.permute(0, 2, 1).contiguous().view(B, C, H, W)

        decA_2, decB_2 = self.information_interaction(decA_2, decB_2)
        # print(f"decA_2: {decA_2.shape}")
        featC = decB_2 - decA_2

        loss_contrastive = None
        if train_mode and text is not None:
            loss_contrastive = self.contrastive_learning(text, featA_s16, featB_s16)

        featC = self.resCD(featC)
        # print(f"featC: {featC.shape}")
        featC = self.headC(featC) 

        outC = self.segmenterC(featC)
        featC_1 = self.regressor(featC)
        outC3d = self.active3d(featC_1)

        
        
        if train_mode:
            return loss_contrastive, F.interpolate(outC, input_shape, mode="bilinear", align_corners=True),\
                F.interpolate(outC3d, input_shape, mode="bilinear", align_corners=True)
        else:
            return F.interpolate(outC, input_shape, mode="bilinear", align_corners=True),\
                F.interpolate(outC3d, input_shape, mode="bilinear", align_corners=True)
