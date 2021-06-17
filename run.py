import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import os
from types import SimpleNamespace as SN
from VQ_VAE.auto_encoder import VQ_VAE
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import Utils

Image_dir = 'Images'

class TrainImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        filename = self.imgs[index]
        cur_label = filename[0].split('\\')[1]

        return super(TrainImageFolder, self).__getitem__(index)[0], cur_label


if __name__ == '__main__':
    config = Utils.config_copy(Utils.get_config())
    args = SN(**config)
    save_dir = os.getcwd()
    use_cuda = torch.cuda.is_available()
    GPU_indx = 0
    device = torch.device(GPU_indx if use_cuda else "cpu")

   # normalize = transforms.Normalize(mean=[1, 1, 1], std=[0, 0, 0])

    train_loader = data.DataLoader(
        TrainImageFolder(Image_dir,
                         transforms.Compose([
                             transforms.RandomResizedCrop(192),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                         ])),
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    dataiter = iter(train_loader)
    test_images = dataiter.next()[0]

    feature_extractor = Utils.get_feature_extractor(device)
    vae_net = VQ_VAE(3, 10, lr=args.lr).to(device)

    loss_log = []
    Utils.make_safe_dir(save_dir)

    start_epoch, loss_log = Utils.load_check_point(args, save_dir, vae_net)

    recon_data, mu, logvar = vae_net(test_images.to(device))

    for epoch in range(start_epoch, args.nepoch):
        vae_net.lr_Linear(args.nepoch, epoch, args.lr)
        for step, data in enumerate(train_loader, 0):
            cur_data = data[0].to(device)

            loss = vae_net.learn(cur_data)
            loss_log.append(loss)

            if step % 100 == 0 and step is not 0:
                print('Epoch: [%d/%d], step: [%d/%d] loss: %.4f' % (epoch + 1, args.nepoch, step, len(train_loader), loss))


        with torch.no_grad():
            recon_data, _, _ = vae_net(test_images.to(device))

            vutils.save_image(torch.cat((torch.sigmoid(recon_data.cpu()), test_images), 2),
                              "%s/%s/%s_%d.png" % (save_dir, "Results", args.model_name, args.imageSize))

            torch.save({
                'epoch': epoch,
                'loss_log': loss_log,
                'model_state_dict': vae_net.state_dict(),
                'optimizer_state_dict': vae_net.optimizer.state_dict()

            }, save_dir + "/Models/" + args.model_name + "_" + str(args.imageSize) + ".pt")




