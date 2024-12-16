import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

import random
import cv2

# from cldm.model import create_model, load_state_dict
# model = create_model('./models/cldm_v15_inpainting.yaml')
# resume_path =  "/data/2023text2edit/ControlNet/ckpt_inpainting_from5625+5625/epoch0_global-step3750.ckpt"
# model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=True)
# model.half()
# model.cuda()

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="input",
                 output_key="jpg",
                 gray_image_key="gray",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.gray_image_key = gray_image_key
        self.output_key=output_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
        # model = create_model('./models/cldm_v15_inpainting.yaml')
        # resume_path =  "/data/2023text2edit/ControlNet/ckpt_inpainting_from5625+5625/epoch0_global-step3750.ckpt"
        # model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=True)
        # model.half()
        # self.model=model.cuda()
        # # self.model=model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = False

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z,gray_content_z):
        z = self.post_quant_conv(z)
        gray_content_z = self.post_quant_conv(gray_content_z)
        dec = self.decoder(z,gray_content_z)
        return dec

    def forward(self, input,gray_image, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        gray_posterior = self.encode(gray_image)
        if sample_posterior:
            gray_content_z = gray_posterior.sample()
        else:
            gray_content_z = gray_posterior.mode()
        dec = self.decode(z,gray_content_z)
        return dec, posterior

    def get_input(self, batch,k0, k1,k2):
        # print(batch)
        # print(k)
        # x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        gray_image = batch[k2]
        if len(gray_image.shape) == 3:
            gray_image = gray_image[..., None]
        gray_image = gray_image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        
        
        # t=random.randint(1,100)#120
        # print(t)
        # model=model.cuda()
        x = batch[k0]#self.model.get_noised_images(((gt.squeeze(0)+1.0)/2.0).permute(2,0,1).to(memory_format=torch.contiguous_format).type(torch.HalfTensor).cuda(),t=torch.Tensor([t]).long().cuda())
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        # x = x.float()
        # torch.cuda.empty_cache()
        # print(input.shape)
        # cv2.imwrite("tttt.png",cv2.cvtColor(x.squeeze(0).permute(1,2,0).cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # x = x*2.0-1.0
        # x = x.squeeze(0).permute(1,2,0).cpu().numpy()*2.0-1.0
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        gt = batch[k1]
        if len(gt.shape) == 3:
            gt = gt[..., None]
        gt = gt.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        
        return gt,x,gray_image

    def training_step(self, batch, batch_idx, optimizer_idx):
        with torch.no_grad():
            outputs,inputs,gray_images = self.get_input(batch, self.image_key,self.output_key,self.gray_image_key)
        reconstructions, posterior = self(inputs,gray_images)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(outputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print(aeloss)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(outputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print(discloss)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        outputs,inputs,gray_images = self.get_input(batch, self.image_key,self.output_key,self.gray_image_key)
        reconstructions, posterior = self(inputs,gray_images)
        aeloss, log_dict_ae = self.loss(outputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+postfix)

        discloss, log_dict_disc = self.loss(outputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val"+postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        # ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
        #     self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        # for name,param in self.decoder.named_parameters():
        #     if "dcn" in name:
        #         print(name)
        ae_params_list = list(self.decoder.dcn_in.parameters())+list(self.decoder.mid.block_1.dcn1.parameters())+list(self.decoder.mid.block_1.dcn2.parameters())+list(self.decoder.mid.block_2.dcn1.parameters())+list(self.decoder.mid.block_2.dcn2.parameters())
        # print(ae_params_list)
        # for i in ae_params_list:
        #     print(i)
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def get_gray_content_z(self,gray_image):
        # if len(gray_image.shape) == 3:
        #     gray_image = gray_image[..., None]
        gray_image = gray_image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        gray_content_z=self.encode(gray_image)
        gray_content_z = gray_content_z.sample()
        return gray_content_z
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        gt,x,gray_image = self.get_input(batch, self.image_key,self.output_key,self.gray_image_key)
        log['gt']=gt
        x = x.to(self.device)
        gray_image = gray_image.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x,gray_image)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gray_image = self.to_rgb(gray_image)
                xrec = self.to_rgb(xrec)
            gray_content_z=self.encode(gray_image)
            gray_content_z = gray_content_z.sample()
            log["samples"] = self.decode(torch.randn_like(posterior.sample()),gray_content_z)
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

