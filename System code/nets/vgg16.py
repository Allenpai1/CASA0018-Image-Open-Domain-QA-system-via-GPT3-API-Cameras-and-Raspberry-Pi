import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


#--------------------------------------#
#   VGG16 network architecture
#--------------------------------------#
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #--------------------------------------#
        #   average pooling 7x7 size
        #--------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #--------------------------------------#
        #   classification part
        #--------------------------------------#
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #--------------------------------------#
        #   feature extraction
        #--------------------------------------#
        x = self.features(x)
        #--------------------------------------#
        #   average pooling
        #--------------------------------------#
        x = self.avgpool(x)
        #--------------------------------------#
        #   flatten
        #--------------------------------------#
        x = torch.flatten(x, 1)
        #--------------------------------------#
        #   classification
        #--------------------------------------#
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

'''
Suppose input images are (600, 600, 3)，with cfg loop，the feature layer size are changed as below：
600,600,3 -> 600,600,64 -> 600,600,64 -> 300,300,64 -> 300,300,128 -> 300,300,128 -> 150,150,128 -> 150,150,256 -> 150,150,256 -> 150,150,256 
-> 75,75,256 -> 75,75,512 -> 75,75,512 -> 75,75,512 -> 37,37,512 ->  37,37,512 -> 37,37,512 -> 37,37,512
when end cfg，we get a 37,37,512 feature map/layer
'''

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

#--------------------------------------#
#   feature extraction parts
#--------------------------------------#
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def decom_vgg16(pretrained = False):
    model = VGG(make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   get feature map with size 37,37,1024
    #----------------------------------------------------------------------------#
    features    = list(model.features)[:30]
    #----------------------------------------------------------------------------#
    #   get the classification feature map with Dropout part remove
    #----------------------------------------------------------------------------#
    classifier  = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]

    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier
