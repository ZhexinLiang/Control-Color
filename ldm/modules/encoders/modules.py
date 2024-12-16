import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, AutoProcessor, CLIPVisionModel, CLIPImageProcessor

import open_clip
from ldm.util import default, count_params
import kornia
# import clip
from einops import rearrange

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        # print(z.shape)
        return z

    def encode(self, text):
        return self(text)

# class FrozenCLIPDualEmbedder(AbstractEncoder):
#     """Uses the CLIP transformer encoder for text (from huggingface)"""
#     LAYERS = [
#         "last",
#         "pooled",
#         "hidden"
#     ]
#     def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
#                  freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
#         super().__init__()
#         assert layer in self.LAYERS
#         self.tokenizer = CLIPTokenizer.from_pretrained(version)
#         self.transformer = CLIPTextModel.from_pretrained(version)
#         # self.processor = CLIPImageProcessor.from_pretrained(version)
#         # self.imagetransformer = CLIPVisionModel.from_pretrained(version)
#         self.ImageEmbedder=FrozenClipImageEmbedder()
#         self.device = device
#         self.max_length = max_length
#         if freeze:
#             self.freeze()
#         self.layer = layer
#         self.layer_idx = layer_idx
#         if layer == "hidden":
#             assert layer_idx is not None
#             assert 0 <= abs(layer_idx) <= 12

#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         #self.train = disabled_train
#         for name,param in self.named_parameters():
#             if not "imagetransformer" in name and not "imageconv" in name and not "ImageEmbedder" in name:
#                 # print(name,param)
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
#                 # print(name)

#     def forward(self, text):
#         # print("text:",len(text))
#         # if len(text)==1:
#         #     txt=text[0]
#         #     hint_image=None
#         # elif len(text)==2:
#         #     txt,hint_image=text
#         txt,hint_image=text
#         # print(hint_image.shape)
#         batch_encoding = self.tokenizer(txt, truncation=True, max_length=self.max_length, return_length=True,
#                                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
#         tokens = batch_encoding["input_ids"].to(self.device)
#         outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
#         # input_image_batch_encoding = self.processor(input_image,return_tensors="pt")
#         # ii_tokens = input_image_batch_encoding["input_ids"].to(self.device)
#         # ii_outputs = self.imagetransformer(input_ids=ii_tokens, output_hidden_states=self.layer=="hidden")
        
#         # hint_image_batch_encoding = self.processor(hint_image,return_tensors="pt")
#         # hi_tokens = hint_image_batch_encoding["input_ids"].to(self.device)
#         # hi_outputs = self.imagetransformer(input_ids=hi_tokens, output_hidden_states=self.layer=="hidden")
        
#         # hint_outputs = hi_outputs-ii_outputs
#         # if hint_image==None:
#         #     if self.layer == "last":
#         #         z = outputs.last_hidden_state
#         #     elif self.layer == "pooled":
#         #         z = outputs.pooler_output[:, None, :]
#         #     else:
#         #         z = outputs.hidden_states[self.layer_idx]
#         #     # print("z",z.shape)
#         #     return z
#         hint_outputs=self.ImageEmbedder(hint_image)
#         # print("hint_outputs",hint_outputs.shape)
#         # print("prompt",outputs.last_hidden_state.shape)
#         if self.layer == "last":
#             z = torch.cat((outputs.last_hidden_state,hint_outputs.unsqueeze(0)),1)#torch.cat((outputs.last_hidden_state,hint_outputs.last_hidden_state),1)#torch.cat((outputs.last_hidden_state,hint_outputs.unsqueeze(0)),1)
#         elif self.layer == "pooled":
#             z = torch.cat((outputs.pooler_output[:, None, :],hint_outputs.unsqueeze(0)),1)
#         else:
#             z = torch.cat((outputs.hidden_states[self.layer_idx],hint_outputs.unsqueeze(0)),1)
#         # print("z",z.shape)
#         return z

#     def encode(self, text):
#         # print(text.shape)
#         return self(text)
    
    
class FrozenCLIPDualEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        # self.processor = CLIPImageProcessor.from_pretrained(version)
        # self.imagetransformer = CLIPVisionModel.from_pretrained(version)
        self.ImageEmbedder=FrozenClipImageEmbedder()
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
        print("pooled")

    def freeze(self):
        # self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for name,param in self.named_parameters():
            # print(name)
            # if not "imagetransformer" in name and not "imageconv" in name and not "ImageEmbedder" in name:
            param.requires_grad = False
            # if not "ImageEmbedder" in name:
            #     # print(name,param)
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = True
        
  
    def forward(self, text):
        # pdb.set_trace()
        # print("text:",len(text))
        # if len(text)==1:
        #     txt=text[0]
        #     hint_image=None
        # elif len(text)==2:
        #     txt,hint_image=text
        txt,hint_image=text
        # if hint_image==None:
        #     batch_encoding = self.tokenizer(txt, truncation=True, max_length=self.max_length, return_length=True,
        #                                     return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        #     tokens = batch_encoding["input_ids"].to(self.device)

        #     outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        #     prompt_outputs=outputs.last_hidden_state
        #     return prompt_outputs
        # else:
        # hint_image.requires_grad_(True)
        # print(hint_image.shape)
        batch_encoding = self.tokenizer(txt, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        prompt_outputs=outputs.last_hidden_state
        # prompt_outputs=outputs.last_hidden_state.detach().requires_grad_(True)
        # prompt_outputs.retain_grad()
        # input_image_batch_encoding = self.processor(input_image,return_tensors="pt")
        # ii_tokens = input_image_batch_encoding["input_ids"].to(self.device)
        # ii_outputs = self.imagetransformer(input_ids=ii_tokens, output_hidden_states=self.layer=="hidden")
        
        # hint_image_batch_encoding = self.processor(hint_image,return_tensors="pt")
        # hi_tokens = hint_image_batch_encoding["input_ids"].to(self.device)
        # hi_outputs = self.imagetransformer(input_ids=hi_tokens, output_hidden_states=self.layer=="hidden")
        
        # hint_outputs = hi_outputs-ii_outputs
        # if hint_image==None:
        #     if self.layer == "last":
        #         z = outputs.last_hidden_state
        #     elif self.layer == "pooled":
        #         z = outputs.pooler_output[:, None, :]
        #     else:
        #         z = outputs.hidden_states[self.layer_idx]
        #     # print("z",z.shape)
        #     return z
        # pdb.set_trace()
        outputs = self.ImageEmbedder(hint_image)
        # image_embeds = outputs.pooler_output #outputs.image_embeds
        image_embeds = outputs.pooler_output
        # print(image_embeds.shape)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output
        # print("hint_outputs",last_hidden_state.shape)
        # print("pooled_output", pooled_output.shape)
        # print("prompt",prompt_outputs.shape)
        
        if self.layer == "last":
            # print(prompt_outputs.shape)
            # print(image_embeds.shape)
            z = torch.cat((prompt_outputs,image_embeds.unsqueeze(1)),1)#,hint_outputs.unsqueeze(0)),1)
            # z = torch.cat((prompt_outputs,hint_outputs.last_hidden_state),1)#,hint_outputs.unsqueeze(0)),1)
        elif self.layer == "pooled":
            z = torch.cat((outputs.pooler_output[:, None, :],hint_outputs.unsqueeze(0)),1)
        else:
            z = torch.cat((outputs.hidden_states[self.layer_idx],hint_outputs.unsqueeze(0)),1)
    
        return z
    # def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
    #              freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
    #     super().__init__()
    #     assert layer in self.LAYERS
    #     # self.processor = CLIPImageProcessor.from_pretrained(version)
    #     # self.imagetransformer = CLIPVisionModel.from_pretrained(version)
    #     self.ImageEmbedder=FrozenClipImageEmbedder()
    #     self.device = device
    #     self.max_length = max_length
    #     if freeze:
    #         self.freeze()
    #     self.layer = layer
    #     self.layer_idx = layer_idx
    #     if layer == "hidden":
    #         assert layer_idx is not None
    #         assert 0 <= abs(layer_idx) <= 12

    # def freeze(self):
    #     #self.train = disabled_train
    #     for name,param in self.named_parameters():
    #         if not "imagetransformer" in name and not "imageconv" in name and not "ImageEmbedder" in name:
    #             # print(name,param)
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True
    #             # print(name)
    
    # def forward(self, txt,hint_image):
    #     # pdb.set_trace()
    #     hint_outputs=self.ImageEmbedder(hint_image)
    #     # print("hint_outputs",hint_outputs.shape)
    #     # print("prompt",outputs.last_hidden_state.shape)
    #     if self.layer == "last":
    #         print(txt.shape)
    #         print(hint_outputs.last_hidden_state.shape)
    #         z = torch.cat((txt,hint_outputs.last_hidden_state),1)#,hint_outputs.unsqueeze(0)),1)
    #     elif self.layer == "pooled":
    #         z = torch.cat((txt,hint_outputs.unsqueeze(0)),1)
    #     else:
    #         z = torch.cat((txt,hint_outputs.unsqueeze(0)),1)
    #     # print("z",z.shape)
    #     return z
    
    def encode(self, text):

        # if isinstance(text, dict):
        #     txt,hint_image=text['c_crossattn'][0]
        #     txt=txt
        # else:
        #     txt,hint_image=text
        # txt = text
        txt, x = text 
        # if x==None:
        #     return self((txt,x))
        # print(x.shape)
        if len(x.shape) == 3:
            x = x[..., None]
        
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        x = [x[i] for i in range(x.shape[0])]
        return self((txt, x))
    
class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]

class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model='ViT-B/16', #ViT-L/14
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        # self.model, _ = clip.load(name=model, device=device, jit=jit)
        # self.model.requires_grad_(True)
        self.imageconv = nn.Conv2d(4,3,(3,3),padding=1)#.cuda()
        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.imagetransformer = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")


    # def preprocess(self, x):
    #     # normalize to [0,1]
    #     # print(x.shape)
    #     # pdb.set_trace()
    #     x = kornia.geometry.resize(x, (224, 224),
    #                                interpolation='bicubic',align_corners=True,
    #                                antialias=self.antialias)
    #     # print("after",x.shape)
    #     # x = (x + 1.) / 2.
    #     print(x)
    #     # renormalize according to clip
    #     x = kornia.enhance.normalize(x, self.mean, self.std)
    #     # print("after1111111",x.shape)
    #     return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        # pdb.set_trace()
        # with torch.set_grad_enabled(True):
        #     print("before",x.shape)
        #     x=self.imageconv(x)
        #     print("after",x.shape)
        # x = x.tolist()
        
        x = self.processor(x, return_tensors="pt")
        # print(x)
        # pdb.set_trace()
        x['pixel_values'] = x['pixel_values'].to(self.device)
        outputs = self.model(**x)
        return outputs
    
# class FrozenClipImageEmbedder(nn.Module):
#     """
#         Uses the CLIP image encoder.
#         """
#     def __init__(
#             self,
#             model='ViT-B/16',
#             jit=False,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#             antialias=False,
#         ):
#         super().__init__()
#         self.model, _ = clip.load(name=model, device=device, jit=jit)
#         # self.imageconv = nn.Conv2d(4,3,(3,3),stride=2)
#         self.antialias = antialias

#         self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
#         self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

#     def preprocess(self, x):
#         # normalize to [0,1]
#         # print(x.shape)
#         x = kornia.geometry.resize(x, (224, 224),
#                                    interpolation='bicubic',align_corners=True,
#                                    antialias=self.antialias)
#         # print("after",x.shape)
#         x = (x + 1.) / 2.
#         # renormalize according to clip
#         x = kornia.enhance.normalize(x, self.mean, self.std)
#         # print("after1111111",x.shape)
#         return x

#     def forward(self, x):
#         # x is assumed to be in range [-1,1]
#         # x=self.imageconv(x)
#         return self.model.encode_image(self.preprocess(x))

# class FrozenClipImageEmbedder(nn.Module):
#     """
#         Uses the CLIP image encoder.
#         """
#     def __init__(
#             self,
#             model='ViT-B/16', #ViT-L/14
#             jit=False,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#             antialias=False,
#         ):
#         super().__init__()
#         self.model, _ = clip.load(name=model, device=device, jit=jit)
#         # self.model.requires_grad_(True)
#         # self.imageconv = nn.Conv2d(4,3,(3,3),padding=1)#.cuda()#padding=1 #stride=2
#         self.antialias = antialias

#         self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
#         self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        
#         # self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
#         self.imagetransformer = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

#     def preprocess(self, x):
#         # normalize to [0,1]
#         # print(x.shape)
#         # pdb.set_trace()
#         x = kornia.geometry.resize(x, (224, 224),
#                                    interpolation='bicubic',align_corners=True,
#                                    antialias=self.antialias)
#         # print("after",x.shape)
#         x = (x + 1.) / 2.
#         # renormalize according to clip
#         x = kornia.enhance.normalize(x, self.mean, self.std)
#         # print("after1111111",x.shape)
#         return x

#     def forward(self, x):
#         # x is assumed to be in range [-1,1]
#         # x=self.imageconv(x)
#         return self.imagetransformer(self.preprocess(x), output_hidden_states="last"=="hidden") #self.model.encode_image(self.preprocess(x))
