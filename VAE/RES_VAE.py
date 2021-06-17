import torch.nn as nn
from VAE.modules import Encoder
from VAE.modules import Decoder
import torch.nn.functional as F
import torch.optim as optim
import Utils


class res_vae(nn.Module):
    def __init__(self, channel_in, feature_extractor, lr, z=512):
        super(res_vae, self).__init__()

        self.encoder = Encoder(channel_in, z=z)
        self.decoder = Decoder(channel_in, z=z)

        self.feature_extractor = feature_extractor

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))

    def forward(self, x, Train=True):
        encoding, mu, logvar = self.encoder(x, Train)
        recon = self.decoder(encoding)
        return recon, mu, logvar

    def vae_loss(self, recon, x, mu, logvar):
        recon_loss = F.binary_cross_entropy_with_logits(recon, x)
        KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = recon_loss + 0.01 * KL_loss
        return loss

    # Linear scaling the learning rate down
    def lr_Linear(self, epoch_max, epoch, lr):
        lr_adj = ((epoch_max - epoch) / epoch_max) * lr
        self.set_lr(lr=lr_adj)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_optimizer(self, check_point_data):
        self.optimizer.load_state_dict(check_point_data)

    def learn(self, data):
        recon_data, mu, logvar = self.forward(data)

        # VAE loss
        loss = self.vae_loss(recon_data, data, mu, logvar)

        # Perception loss
       # loss_feature = Utils.feature_loss(data, recon_data, self.feature_extractor)

        #loss += loss_feature

        self.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
