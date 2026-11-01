import spaces
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import torchvision.transforms.functional as TF
import gradio as gr
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading Stable Video Diffusion model...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=dtype,
    variant="fp16"
)
pipe.to(device)
pipe.enable_attention_slicing()

def pytorch_linear_interpolate(frame_list):
    smoothed_frames = []
    for i in range(len(frame_list) - 1):
        smoothed_frames.append(frame_list[i])
        img1 = TF.to_tensor(frame_list[i])
        img2 = TF.to_tensor(frame_list[i+1])
        mid_tensor = (img1 + img2) / 2.0
        smoothed_frames.append(TF.to_pil_image(mid_tensor))
    smoothed_frames.append(frame_list[-1])
    return smoothed_frames

@spaces.GPU(duration=120) 
def animate_image(image, motion_bucket_id, fps_val, num_frames_val, use_interpolation):
    if image is None:
        return "Please upload an image first."
    image = image.convert("RGB").resize((1024, 576))
    
    print("Generating base animation frames on ZeroGPU...")
    generator = torch.manual_seed(42)
    frames = pipe(
        image, 
        decode_chunk_size=8, 
        generator=generator,
        motion_bucket_id=int(motion_bucket_id),
        fps=int(fps_val),
        num_frames=int(num_frames_val)
    ).frames[0]
    
    final_fps = int(fps_val)
    if use_interpolation:
        frames = pytorch_linear_interpolate(frames)
        final_fps = int(fps_val) * 2

    output_path = "output_animation_smooth.mp4"
    export_to_video(frames, output_path, fps=final_fps)
    return output_path

with gr.Blocks(title="Cloud AI Image Animator") as demo:
    gr.Markdown("# 🎬 Hugging Face ZeroGPU Image-to-Animation Studio (SVD)")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Source Image")
            motion_slider = gr.Slider(minimum=50, maximum=255, value=127, step=1, label="Motion Bucket ID")
            fps_slider = gr.Slider(minimum=4, maximum=12, value=7, step=1, label="Base FPS")
            frames_slider = gr.Slider(minimum=14, maximum=25, value=14, step=1, label="Number of Base Frames")
            interp_checkbox = gr.Checkbox(value=True, label="Enable Smooth Frame Blending")
            submit_btn = gr.Button("Generate Smooth Animation 🚀", variant="primary")
        with gr.Column():
            output_video = gr.Video(label="Generated Smooth Animation")
            
    submit_btn.click(
        fn=animate_image,
        inputs=[input_image, motion_slider, fps_slider, frames_slider, interp_checkbox],
        outputs=output_video
    )

demo.launch()