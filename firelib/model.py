import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision

import timm



class FireModel(nn.Module):
    def __init__(self, cfg):
        super(FireModel, self).__init__()

        self.cfg = cfg
        

        self.init_model()
        
        self.update_model_structure()
        


    def init_model(self):

        if 'timm:' in self.cfg['model_name']: #efficientnetv2 , eca_nfnet_l0, convnextv2, convnext, swin, resnet,, xception
            pretrained = False
            pretrained_cfg_overlay = None
            features_only = False


            if self.cfg['pretrained']:
                pretrained = True
                if self.cfg['pretrained'] != 'default':
                    pretrained_cfg_overlay = dict(file=self.cfg['pretrained'])
            if self.cfg['head_type']:
                features_only = True

            #print(pretrained)
            if self.cfg['pretrained'] in ['default', '']:
                self.backbone = timm.create_model(self.cfg['model_name'].replace('timm:',''), 
                                                        in_chans=3,
                                                        features_only=features_only,
                                                        pretrained=pretrained,
                                                        num_classes=self.cfg['num_classes'],
                                                        pretrained_cfg_overlay=dict(file=self.cfg['pretrained']))
  


        # [Add new model here]
        # elif self.cfg['model_name']=="xxx":
        #     pass


        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


    def update_model_structure(self):
        
        if 'timm:' in self.cfg['model_name']:
            pass  
            
        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])


        if self.cfg['head_type']:
            self.head = self.get_head()

        
    def get_head(self):
        if self.cfg['head_type'] == 'cls':
            return nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(320, self.cfg['num_classes'])
                            )

        else:
            raise Exception("[ERROR] Unknown head_type: ",self.cfg['head_type'])


    def forward(self, img):        

        if 'timm:' in self.cfg['model_name']:
            out1 = self.backbone(img)
            # print([x.shape for x in out1])

            if self.cfg['head_type'] == 'cls':
                out1 = self.head(out1[-1])

            out = [out1]


        else:
            raise Exception("[ERROR] Unknown model_name: ",self.cfg['model_name'])

        return out


