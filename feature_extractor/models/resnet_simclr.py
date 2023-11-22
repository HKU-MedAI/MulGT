import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1, out_3


class KimiaNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(KimiaNetSimCLR, self).__init__()
        self.kimia_dict = {"kimianet": models.densenet121(pretrained=True)}

        resnet18 = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)

        model = self._get_basemodel(base_model)

        model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        num_ftrs = model.classifier.in_features
        model_final = fully_connected(model.features, num_ftrs, 30)
        KimiaNetPyTorchWeights_path = "KimiaNetPyTorchWeights.pth"

        checkpoint = torch.load(KimiaNetPyTorchWeights_path)
        new_state_dict = collections.OrderedDict()

        for k, v in checkpoint.items():
            name = k[7:]  # remove "module."
            new_state_dict[name] = v

        model_final.load_state_dict(new_state_dict)

        num_ftrs = model_final.fc_4.in_features

        self.features = nn.Sequential(*list(model_final.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.kimia_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
