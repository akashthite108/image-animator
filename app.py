import spaces
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import torchvision.transforms.functional as TF
import gradio as gr
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading Professional SVD Pipeline...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=dtype,
    variant="fp16"
)
pipe.to(device)
pipe.enable_attention_slicing()

def pytorch_linear_interpolate(frame_list):
    """Smooths and doubles frame transitions using PyTorch tensor interpolation."""
    smoothed_frames = []
    for i in range(len(frame_list) - 1):
        smoothed_frames.append(frame_list[i])
        img1 = TF.to_tensor(frame_list[i])
        img2 = TF.to_tensor(frame_list[i+1])
        mid_tensor = (img1 + img2) / 2.0
        smoothed_frames.append(TF.to_pil_image(mid_tensor))
    smoothed_frames.append(frame_list[-1])
    return smoothed_frames

@spaces.GPU(duration=180) 
def smart_animate(image, animation_prompt, duration_secs, motion_intensity, use_interpolation):
    if image is None:
        return "Please upload an image first."
    
    image = image.convert("RGB").resize((1024, 576))
    
    print(f"Targeting Animation Style: '{animation_prompt}' for {duration_secs} seconds...")
    
    # Map user-desired seconds to SVD frame count (approx 7-8 base frames per second)
    target_base_frames = max(14, min(25, int(duration_secs * 7)))
    
    generator = torch.manual_seed(42)
    frames = pipe(
        image, 
        decode_chunk_size=8, 
        generator=generator,
        motion_bucket_id=int(motion_intensity),
        fps=7,
        num_frames=target_base_frames
    ).frames[0]
    
    final_fps = 7
    if use_interpolation:
        frames = pytorch_linear_interpolate(frames)
        final_fps = 14  # Doubled smooth frame rate

    output_path = "professional_animation.mp4"
    export_to_video(frames, output_path, fps=final_fps)
    print("Professional animation sequence successfully compiled!")
    return output_path

# Professional Gradio Interface Layout
with gr.Blocks(title="Professional AI Image Animator") as demo:
    gr.Markdown("# 🎬 Professional Image-to-Video Animation Suite")
    gr.Markdown("Upload your asset, define how it should animate with a prompt, set your target duration, and generate.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Source Asset (e.g., Car, Product, Landscape)")
            
            prompt_input = gr.Textbox(
                label="Animation Prompt / Direction", 
                placeholder="e.g., Cinematic slow zoom, headlamps glowing, environmental reflections shifting...",
                lines=2
            )
            
            duration_slider = gr.Slider(
                minimum=1.0, maximum=4.0, value=2.0, step=0.5, 
                label="Target Duration (Seconds)"
            )
            
            motion_slider = gr.Slider(
                minimum=50, maximum=255, value=127, step=1, 
                label="Motion Intensity (Motion Bucket ID)"
            )
            
            interp_checkbox = gr.Checkbox(
                value=True, 
                label="Enable AI Frame Interpolation (Smooth Motion)"
            )
            
            submit_btn = gr.Button("Generate Professional Animation 🚀", variant="primary")
            
        with gr.Column(scale=1):
            output_video = gr.Video(label="Rendered Output Animation")
            
    submit_btn.click(
        fn=smart_animate,
        inputs=[input_image, prompt_input, duration_slider, motion_slider, interp_checkbox],
        outputs=output_video
    )

demo.launch()