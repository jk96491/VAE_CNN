import os
import torch
from copy import deepcopy
import yaml
from GetFeatures import GetFeatures
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
import torchvision.models as models
import torch.nn as nn


def get_data_STL10(transform, batch_size, download=True, root="/data"):
    print("Loading trainset...")
    trainset = Datasets.STL10(root=root, split='unlabeled', transform=transform, download=download)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Loading testset...")
    testset = Datasets.STL10(root=root, split='test', download=download, transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Done!")

    return trainloader, testloader


def make_safe_dir(save_dir):
    if not os.path.isdir(save_dir + "/Models"):
        os.makedirs(save_dir + "/Models")
    if not os.path.isdir(save_dir + "/Results"):
        os.makedirs(save_dir + "/Results")


def load_check_point(args, save_dir, vae_net):
    start_epoch = 0
    loss_log = []
    if args.load_checkpoint:
        checkpoint = torch.load(save_dir + "/Models/" + args.model_name + "_" + str(args.imageSize) + ".pt", map_location="cpu")
        print("Checkpoint loaded")
        vae_net.set_optimizer(checkpoint['optimizer_state_dict'])
        vae_net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint["epoch"]
        loss_log = checkpoint["loss_log"]
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        if os.path.isfile(save_dir + "/Models/" + args.model_name + "_" + str(args.imageSize) + ".pt"):
            raise ValueError("Warning Checkpoint exists")
        else:
            print("Starting from scratch")

    return start_epoch, loss_log


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def get_config():
    config_dir = '{0}/{1}'

    with open(config_dir.format('Config', "{}.yaml".format('default')), "r") as f:
        try:
            default_config = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    final_config_dict = default_config

    return final_config_dict


def get_feature_extractor(device, layers_deep=7):
    C_net = models.vgg19(pretrained=True).to(device)
    C_net = C_net.eval()

    layers = []
    for i in range(layers_deep):
        layers.append(C_net.features[i])
        if isinstance(C_net.features[i], nn.ReLU):
            layers.append(GetFeatures())
    return nn.Sequential(*layers)


def feature_loss(img, recon_data, feature_extractor):
    img_cat = torch.cat((img, torch.sigmoid(recon_data)), 0)
    out = feature_extractor(img_cat)
    loss = 0
    for i in range(len(feature_extractor)):
        if isinstance(feature_extractor[i], GetFeatures):
            loss += (feature_extractor[i].features[:(img.shape[0])] - feature_extractor[i].features[
                                                                      (img.shape[0]):]).pow(2).mean()
    return loss / (i + 1)

