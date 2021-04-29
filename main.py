import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import os
from types import SimpleNamespace as SN
from VAE.RES_VAE import res_vae
import Utils

# 기본 세팅
config = Utils.config_copy(Utils.get_config())
args = SN(**config)
save_dir = os.getcwd()
use_cuda = torch.cuda.is_available()
GPU_indx  = 0
device = torch.device(GPU_indx if use_cuda else "cpu")

# Data Load
transform = T.Compose([T.Resize(args.imageSize), T.ToTensor()])
trainloader, testloader = Utils.get_data_STL10(transform, args.batchSize, download=False, root=args.dataset_root)

dataiter = iter(testloader)
test_images = dataiter.next()[0]

feature_extractor = Utils.get_feature_extractor(device)
vae_net = res_vae(channel_in=3, feature_extractor=feature_extractor, lr=args.lr).to(device)

loss_log = []
Utils.make_safe_dir(save_dir)

start_epoch, loss_log = Utils.load_check_point(args, save_dir, vae_net)

recon_data, mu, logvar = vae_net(test_images.to(device))

for epoch in range(start_epoch, args.nepoch):
    vae_net.lr_Linear(args.nepoch, epoch, args.lr)
    for step, data in enumerate(trainloader, 0):
        cur_data = data[0].to(device)

        loss = vae_net.learn(cur_data)
        loss_log.append(loss)

        if step % 100 == 0 and step is not 0:
            print('Epoch: [%d/%d], step: [%d/%d] loss: %.4f' % (epoch + 1, args.nepoch, step, len(trainloader), loss))

    with torch.no_grad():
        recon_data, _, _ = vae_net(test_images.to(device), Train=False)

        vutils.save_image(torch.cat((torch.sigmoid(recon_data.cpu()), test_images), 2),
                          "%s/%s/%s_%d.png" % (save_dir, "Results", args.model_name, args.imageSize))

        torch.save({
            'epoch': epoch,
            'loss_log': loss_log,
            'model_state_dict': vae_net.state_dict(),
            'optimizer_state_dict': vae_net.optimizer.state_dict()

        }, save_dir + "/Models/" + args.model_name + "_" + str(args.imageSize) + ".pt")




