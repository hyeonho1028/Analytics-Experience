import torch 
import torch.nn as nn
import torchvision.models as models
import timm

def build_model(args, device):
    model = timm_models(args)
    model = model.to(device)
    return model


class timm_models(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_list = list()
        # model_list.append(nn.Conv2d(1, 3, 1))
        self.model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=512)
        # model_list.append(model)
        # model = nn.Sequential(*model_list)
        # self.model = model
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.output_layer = nn.Linear(512, args.num_classes)

    def extract(self, x):
        x=self.model(x)
        return x

    def forward(self, img):
        feat = self.model(img)
        # for i, dropout in enumerate(self.dropouts):
        #     if i==0:
        #         output = self.output_layer(dropout(feat))
        #     else:
        #         output += self.output_layer(dropout(feat))
        # else:
        #     output /= len(self.dropouts)
        # outputs = torch.sigmoid(output)
        outputs = torch.sigmoid(self.output_layer(feat))
        return outputs