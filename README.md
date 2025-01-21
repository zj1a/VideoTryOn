# Overview
The project uses several open-sourced frameworks (ViViD, LangSAM, and Vid2DensePose) to achieve zero-shot video try-ons with arbitrary clothes and avatars.
For simplicity of this repo, I uploaded most files from [ViViD](https://github.com/alibaba-yuanjing-aigclab/ViViD), and part of the code that I adapted from 
[LangSAM](https://github.com/luca-medeiros/lang-segment-anything) and [Vid2DensePose](https://github.com/Flode-Labs/vid2densepose).

# Dependencies
Please follow the instructions from ViViD (a video diffusion model for virtual try-ons) to install its dependencies. 
Similarly, please follow the repo of LangSAM (which I used to extract segmentation masks for the clothes and the avatar videos) 
and Vid2DensePose (used to generate pose conditions required for ViViD) for additional dependencies.

# Code I adapted
1. ViViD/vivid.py (main entry point)
2. ViViD/lang-segment-anything/segment.py
3. ViViD/vid2densepose/pose.py

# Input data format
1. the input avatar videos should be put in `ViViD/data/videos`
2. the target cloth images should be put in `ViViD/data/cloth`

# Example run
* `cd ViViD`
* `python vivid.py --input_name=example_avatar.mp4 --cloth_name=cloth.jpg --output_name=output.mp4`
* `ffmpeg -i output/output.mp4 -i data/videos/example_avatar.mp4 -map 0:v -map 1:a -c:v copy -shortest output/output_with_audio.mp4`

# Output
The output video will be placed in `ViViD/output` as `output_with_audio.mp4`.
