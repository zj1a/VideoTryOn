# Get the segmentation mask of the cloth region for the input video
# as well as for the target garment image.
def get_masks(input_video_name, input_cloth_name):
    
    import numpy as np
    from PIL import Image
    from lang_sam import LangSAM
    import cv2
    import os

    model = LangSAM()

    # Open the input video
    base_path = '/code/ViViD/data'
    input_video = cv2.VideoCapture(os.path.join(base_path, 'videos', input_video_name))

    # Get the video parameters (frame width, height, fps)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    blur = 5

    # Get segmentation mask for the target cloth.
    cloth_path = os.path.join(base_path, 'cloth', input_cloth_name)
    outputs = model.predict([Image.open(cloth_path).convert('RGB')], ["garment"])
    mask = (outputs[0]['masks'][0] * 255).astype(np.uint8)
    Image.fromarray(mask).convert('RGB').save(
        os.path.join(base_path, 'cloth_mask', input_cloth_name))

    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    output_video = cv2.VideoWriter(
        os.path.join(base_path, 'agnostic_mask', input_video_name), 
            fourcc, fps, (frame_width, frame_height))
    agn_output_video = cv2.VideoWriter(
        os.path.join(base_path, 'agnostic', input_video_name), 
            fourcc, fps, (frame_width, frame_height))

    idx = 0
    while True:
        ret, frame = input_video.read()  # Read a frame from the input video
        if not ret:
            break  # Break the loop if there are no more frames

        # Mask for head area.
        outputs = model.predict(
            [Image.fromarray(frame[..., ::-1]).convert('RGB')], ["head"])[0]
        head_mask = (outputs['masks'][0] * 255).astype(np.uint8)
        head_mask = np.tile(head_mask[..., None], [1, 1, 3])
        blurred_head_mask = cv2.GaussianBlur(head_mask, (35, 35), 0)
        blurred_head_mask = (blurred_head_mask > 0).astype(np.uint8)

        # Mak for upper body.
        outputs = model.predict(
            [Image.fromarray(frame[..., ::-1]).convert('RGB')], ["upper body"])[0]
        mask = (outputs['masks'][0] * 255).astype(np.uint8)
        mask = np.tile(mask[..., None], [1, 1, 3])

        # Apply Gaussian blur to the frame
        blurred_mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        blurred_mask = (blurred_mask > 0).astype(np.uint8) * 255
        
        # Combine the two masks.
        blurred_mask = blurred_mask * (1 - blurred_head_mask)

        # Write the blurred frame to the output video
        output_video.write(blurred_mask)

        blurred_mask = (blurred_mask / 255).astype(np.uint8)
        agn_frame = blurred_mask * 127 + (1 - blurred_mask) * frame
        agn_output_video.write(agn_frame)

    # Release the input and output video objects
    input_video.release()
    output_video.release()
    agn_output_video.release()

    # Optionally, close any OpenCV windows
    cv2.destroyAllWindows()
