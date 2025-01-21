import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="./configs/prompts/upper1.yaml")

    # Input video filename.
    parser.add_argument("--input_name",type=str,default="example_avatar.mp4")
    # Output video filename.
    parser.add_argument("--output_name",type=str,default="output.mp4")
    # Input cloth filename.
    parser.add_argument("--cloth_name",type=str,default="cloth.jpg")
    parser.add_argument("-W", type=int, default=384)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args


def videotryon():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        subfolder="unet",
        unet_additional_kwargs={
            "in_channels": 5,
        }
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    seed = config.get("seed",args.seed)
    generator = torch.manual_seed(seed)

    width, height = args.W, args.H
    clip_length = config.get("L",args.L)  
    steps = args.steps
    guidance_scale = args.cfg

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    base_path = '/code/ViViD/data'
    model_image_path = os.path.join(base_path, 'videos', args.input_name)

    transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    src_fps = get_fps(model_image_path)
    model_name = Path(model_image_path).stem
    agnostic_path=model_image_path.replace("videos","agnostic")
    agn_mask_path=model_image_path.replace("videos","agnostic_mask")
    densepose_path=model_image_path.replace("videos","densepose")

    agnostic_images=read_frames(agnostic_path)
    agn_mask_images=read_frames(agn_mask_path)
    pose_images=read_frames(densepose_path)
    total_masks = (np.stack([np.array(i) for i in agn_mask_images], 0) > 0).astype(np.uint8)
    total_masks = torch.from_numpy(total_masks)[None].permute(0, 4, 1, 2, 3)
    original_video = np.stack([np.array(i) for i in agnostic_images], 0).astype(np.uint8)
    original_video = torch.from_numpy(original_video)[None].permute(0, 4, 1, 2, 3)

    video_list = []

    # Each time we process 24 * 8 frames.
    for i in range(0, clip_length, 8*24):

        agnostic_list=[]
        for agnostic_image_pil in agnostic_images[i:i+8*24]:
            agnostic_list.append(agnostic_image_pil)

        agn_mask_list=[]
        for agn_mask_image_pil in agn_mask_images[i:i+8*24]:
            agn_mask_list.append(agn_mask_image_pil)

        pose_list=[]
        for pose_image_pil in pose_images[i:i+8*24]:
            pose_list.append(pose_image_pil)

        cloth_image_path = os.path.join(base_path, 'cloth', args.cloth_name)
        cloth_image_pil = Image.open(cloth_image_path).convert("RGB")

        cloth_mask_path = cloth_image_path.replace("cloth","cloth_mask")
        cloth_mask_pil = Image.open(cloth_mask_path).convert("RGB")

        pipeline_output = pipe(
            agnostic_list,
            agn_mask_list,
            cloth_image_pil,
            cloth_mask_pil,
            pose_list,
            width,
            height,
            8*24,
            steps,
            guidance_scale,
            generator=generator,
        )
        video = pipeline_output.videos
        video_list.append(video)
    video = torch.cat(video_list, 2)
    
    # Post-processing (optional).
    total_masks = total_masks[:, :, :video.size(2)]
    original_video = original_video[:, :, :video.size(2)].float() / 255
    video = total_masks * video + original_video * (1 - total_masks)

    save_videos_grid(
        video,
        os.path.join('./output', args.output_name),
        n_rows=1,
        fps=src_fps if args.fps is None else args.fps,
    )


if __name__ == "__main__":
    # # STEP I: get masked videos required as inputs to the ViViD model.
    sys.path.append('/code/ViViD/lang-segment-anything')
    from segment import get_masks
    get_masks(args.input_name, args.cloth_name)

    # # STEP II: get pose videos required as inputs to the ViViD model.
    sys.path.append('/code/ViViD/vid2densepose')
    from main import get_densepose
    get_densepose(args.input_name)

    # STEP III: generated cloth try on video with the ViViD model.
    videotryon()