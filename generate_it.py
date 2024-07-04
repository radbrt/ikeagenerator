#%%
from diffusers import DiffusionPipeline
#%%
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl")
#%%
# Generate image
text = "How to assemble a chair"
image = pipeline(text).images[0]
# image = pipeline.generate_images(text, num_images=1)[0]
#%%
# Save image
image.save("chair.jpg")