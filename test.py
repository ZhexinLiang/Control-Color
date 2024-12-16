import os
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image
from cldm.model import create_model, load_state_dict
from cldm.ddim_haced_sag_step import DDIMSampler
from lavis.models import load_model_and_preprocess
from PIL import Image
import tqdm

from ldm.models.autoencoder_train import AutoencoderKL

ckpt_path="./pretrained_models/main_model.ckpt"

model = create_model('./models/cldm_v15_inpainting_infer1.yaml').cpu()
model.load_state_dict(load_state_dict(ckpt_path, location='cuda'),strict=False)
model = model.cuda()

ddim_sampler = DDIMSampler(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLIP_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

vae_model_ckpt_path="./pretrained_models/content-guided_deformable_vae.ckpt"

def load_vae():
    init_config = {
        "embed_dim": 4,
        "monitor": "val/rec_loss",
        "ddconfig":{
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult":[1,2,4,4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        },
        "lossconfig":{
          "target": "ldm.modules.losses.LPIPSWithDiscriminator",
          "params":{
            "disc_start": 501,
            "kl_weight": 0,
            "disc_weight": 0.025,
            "disc_factor": 1.0
        }
        }
    }
    vae = AutoencoderKL(**init_config)
    vae.load_state_dict(load_state_dict(vae_model_ckpt_path, location='cuda'))
    vae = vae.cuda()
    return vae

vae_model=load_vae()

def encode_mask(mask,masked_image):
    mask = torch.nn.functional.interpolate(mask, size=(mask.shape[2] // 8, mask.shape[3] // 8))
    # mask=torch.cat([mask] * 2) #if do_classifier_free_guidance else mask
    mask = mask.to(device="cuda")
    # do_classifier_free_guidance=False
    masked_image_latents = model.get_first_stage_encoding(model.encode_first_stage(masked_image.cuda())).detach()
    return mask,masked_image_latents

def get_mask(input_image,hint_image):
    mask=input_image.copy()
    H,W,C=input_image.shape
    for i in range(H):
        for j in range(W):
            if input_image[i,j,0]==hint_image[i,j,0]:
                # print(input_image[i,j,0])
                mask[i,j,:]=255.
            else:
                mask[i,j,:]=0. #input_image[i,j,:]
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask=cv2.morphologyEx(np.array(mask),cv2.MORPH_OPEN,kernel,iterations=1)
    return mask

def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.
    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.
    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.
    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).
    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

# generate image
generator = torch.manual_seed(859311133)#0
def path2L(img_path):
    raw_image = cv2.imread(img_path)
    raw_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2LAB)
    raw_image_input = cv2.merge([raw_image[:,:,0],raw_image[:,:,0],raw_image[:,:,0]])
    return raw_image_input

def is_gray_scale(img, threshold=10):
    img = Image.fromarray(img)
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False

def randn_tensor(
    shape,
    generator= None,
    device= None,
    dtype=None,
    layout= None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print("The passed generator was created on 'cpu' even though a tensor on {device} was expected.")
                # logger.info(
                #     f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                #     f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                #     f" slighly speed up this function by passing a generator that was created on the {device} device."
                # )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def add_noise(
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        betas = torch.linspace(0.00085, 0.0120, 1000, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

def set_timesteps(num_inference_steps: int, timestep_spacing="leading",device=None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        num_train_timesteps=1000
        if num_inference_steps > num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {num_train_timesteps} timesteps."
            )

        num_inference_steps = num_inference_steps
        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif timestep_spacing == "leading":
            step_ratio = num_train_timesteps // num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            # timesteps += steps_offset
        elif timestep_spacing == "trailing":
            step_ratio = num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        timesteps = torch.from_numpy(timesteps).to(device)
        return timesteps

def get_timesteps(num_inference_steps, timesteps_set, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps_set[t_start * 1 :]

        return timesteps, num_inference_steps - t_start


def get_noised_image_latents(img,W,H,ddim_steps,strength,seed,device):
    img1 = [cv2.resize(img,(W,H))]
    img1 = np.concatenate([i[None, :] for i in img1], axis=0)
    img1 = img1.transpose(0, 3, 1, 2)
    img1 = torch.from_numpy(img1).to(dtype=torch.float32) /127.5 - 1.0
    
    image_latents=model.get_first_stage_encoding(model.encode_first_stage(img1.cuda())).detach()
    shape=image_latents.shape
    generator = torch.manual_seed(seed) 
    
    noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    
    timesteps_set=set_timesteps(ddim_steps,timestep_spacing="linspace", device=device)
    timesteps, num_inference_steps = get_timesteps(ddim_steps, timesteps_set, strength, device)
    latent_timestep = timesteps[1].repeat(1 * 1)

    init_latents = add_noise(image_latents, noise, torch.tensor(latent_timestep))
    for j in range(0, 1000, 100):
        
        x_samples=model.decode_first_stage(add_noise(image_latents, noise, torch.tensor(j)))
        init_image=(einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    
        cv2.imwrite("./initlatents1/"+str(j)+"init_image.png",cv2.cvtColor(init_image[0],cv2.COLOR_RGB2BGR))
    return init_latents
                  
def process(using_deformable_vae,change_according_to_strokes,iterative_editing,input_image,hint_image,prompt, a_prompt, n_prompt, num_samples, image_resolution,  ddim_steps, guess_mode, strength, scale, sag_scale,SAG_influence_step, seed, eta):
    torch.cuda.empty_cache()
    with torch.no_grad():
        ref_flag=True
        input_image_ori=input_image
        if is_gray_scale(input_image):
            print("It is a greyscale image.")
            # mask=get_mask(input_image,hint_image)
        else:
            print("It is a color image.")
            input_image_ori=input_image
            input_image=cv2.cvtColor(input_image,cv2.COLOR_RGB2LAB)[:,:,0]
            input_image=cv2.merge([input_image,input_image,input_image])
        mask=get_mask(input_image_ori,hint_image)
        cv2.imwrite("gradio_mask1.png",mask)
        
        if iterative_editing:
            mask=255-mask
            if change_according_to_strokes:
                hint_image=mask/255.*hint_image+(1-mask/255.)*input_image_ori
            else:
                hint_image=mask/255.*input_image+(1-mask/255.)*input_image_ori
        else:
            hint_image=mask/255.*input_image+(1-mask/255.)*hint_image
        hint_image=hint_image.astype(np.uint8)
        if len(prompt)==0:
            image = Image.fromarray(input_image)
            image = vis_processors["eval"](image).unsqueeze(0).to(device)
            prompt = BLIP_model.generate({"image": image})[0]
            if "a black and white photo of" in prompt or "black and white photograph of" in prompt:
                prompt=prompt.replace(prompt[:prompt.find("of")+3],"")
        print(prompt)
        H_ori,W_ori,C_ori=input_image.shape
        img = resize_image(input_image, image_resolution)
        mask = resize_image(mask, image_resolution)
        hint_image =resize_image(hint_image,image_resolution)
        mask,masked_image=prepare_mask_and_masked_image(Image.fromarray(hint_image),Image.fromarray(mask))
        mask,masked_image_latents=encode_mask(mask,masked_image)
        H, W, C = img.shape
        
        # if ref_image is None:
        ref_image=np.array([[[0]*C]*W]*H).astype(np.float32)
        # print(ref_image.shape)
        # ref_flag=False
        ref_image=resize_image(ref_image,image_resolution)
        
        # cv2.imwrite("exemplar_image.png",cv2.cvtColor(ref_image,cv2.COLOR_RGB2BGR))    
        
        # ddim_steps=1
        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        ref_image=cv2.resize(ref_image,(W,H))
        
        ref_image=torch.from_numpy(ref_image).cuda().unsqueeze(0)
        
        init_latents=None
        
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        print("no reference images, using Frozen encoder")
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(model,ddim_steps, num_samples,
                                                    shape, cond, mask=mask, masked_image_latents=masked_image_latents,verbose=False, eta=eta,
                                                    #  x_T=image_latents,
                                                    x_T=init_latents,
                                                    unconditional_guidance_scale=scale,
                                                    sag_scale = sag_scale,
                                                    SAG_influence_step=SAG_influence_step,
                                                    noise = noise,
                                                    unconditional_conditioning=un_cond)
        

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        if not using_deformable_vae:
            x_samples = model.decode_first_stage(samples)
        else:
            samples = model.decode_first_stage_before_vae(samples)
            gray_content_z=vae_model.get_gray_content_z(torch.from_numpy(img.copy()).float().cuda() / 255.0)
            # print(gray_content_z.shape)
            x_samples = vae_model.decode(samples,gray_content_z)
            
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        #single image replace L channel
        results_ori = [x_samples[i] for i in range(num_samples)]
        results_ori=[cv2.resize(i,(W_ori,H_ori),interpolation=cv2.INTER_LANCZOS4) for i in results_ori]
        
        cv2.imwrite("result_ori.png",cv2.cvtColor(results_ori[0],cv2.COLOR_RGB2BGR))
        
        results_tmp=[cv2.cvtColor(np.array(i),cv2.COLOR_RGB2LAB) for i in results_ori]
        results=[cv2.merge([input_image[:,:,0],tmp[:,:,1],tmp[:,:,2]]) for tmp in results_tmp]
        results_mergeL=[cv2.cvtColor(np.asarray(i),cv2.COLOR_LAB2RGB) for i in results]#cv2.COLOR_LAB2BGR)
        cv2.imwrite("output.png",cv2.cvtColor(results_mergeL[0],cv2.COLOR_RGB2BGR))
    return results_mergeL 

def get_grayscale_img(img, progress=gr.Progress(track_tqdm=True)):
    torch.cuda.empty_cache()
    for j in tqdm.tqdm(range(1),desc="Uploading input..."):
        return img,"Uploading input image done."
    
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control-Color")#("## Color-Anything")#Control Stable Diffusion with L channel
    with gr.Row():
        with gr.Column():
            # input_image = gr.Image(source='upload', type="numpy")
            grayscale_img = gr.Image(visible=False, type="numpy")
            input_image = gr.Image(source='upload',tool='color-sketch',interactive=True)
            Grayscale_button = gr.Button(value="Upload input image")
            text_out = gr.Textbox(value="Please upload input image first, then draw the strokes or input text prompts or give reference images as you wish.")
            prompt = gr.Textbox(label="Prompt")
            change_according_to_strokes = gr.Checkbox(label='Change according to strokes\' color', value=True)
            iterative_editing = gr.Checkbox(label='Only change the strokes\' area', value=False)
            using_deformable_vae = gr.Checkbox(label='Using deformable vae. (Less color overflow)', value=False)
            # with gr.Accordion("Input Reference", open=False):
            #     ref_image = gr.Image(source='upload', type="numpy")
            run_button = gr.Button(label="Upload prompts/strokes (optional) and Run",value="Upload prompts/strokes (optional) and Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                #detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.0, step=0.1)#value=9.0
                sag_scale = gr.Slider(label="SAG Scale", minimum=0.0, maximum=1.0, value=0.05, step=0.01)#0.08
                SAG_influence_step = gr.Slider(label="1000-SAG influence step", minimum=0, maximum=900, value=600, step=50)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)#94433242802
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, detailed, real')#extremely detailed
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='a black and white photo, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            # grayscale_img = gr.Image(interactive=False,visible=False)
           
    Grayscale_button.click(fn=get_grayscale_img,inputs=input_image,outputs=[grayscale_img,text_out])
    ips = [using_deformable_vae,change_according_to_strokes,iterative_editing,grayscale_img,input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale,sag_scale,SAG_influence_step, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0',share=True)
