from torchvision.models import resnet18, resnext50_32x4d, vit_b_16
from torch import nn

MODELS_DICT = {
    'ResNet': resnet18,
    'ResNext': resnext50_32x4d,
    'ViT': vit_b_16
}

UNFREEZE_LAYERS_DICT = {
    'ResNet': ['layer4', 'fc'],
    'ResNext': ['layer4', 'fc'],
    'ViT': ['head']
}

def prep_resnet(model):
    model.fc = nn.Linear(512, 3)
    return model

def prep_resnext(model):
    model.fc = nn.Linear(2048, 3)
    return model

def prep_vit(model):
    model.heads.head = nn.Linear(768, 3)
    return model

PREP_FUNCTIONS_DICT = {
    'ResNet': prep_resnet,
    'ResNext': prep_resnext,
    'ViT': prep_vit
}

def get_model(name):
    architecture = MODELS_DICT[name]
    function = PREP_FUNCTIONS_DICT[name]
    return function(architecture(pretrained=True))

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.model = get_model(config['name'])

        if config['head_only']:
            unfreeze = UNFREEZE_LAYERS_DICT[config['name']]
            for layer_name, layer in self.model.named_parameters():
                for name in unfreeze:
                    if name in layer_name:
                        layer.requires_grad = True
                        break
                    else:
                        layer.requires_grad = False
    
    def forward(self, x):
        return self.model(x)