import torch
from diffusers import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-medium"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = """
A photorealistic smartphone photograph of the inside
of a Japanese home refrigerator.

Visible ingredients:
eggs, tomatoes, cucumbers, carrots, milk.

Realistic refrigerator lighting, natural shadows,
plastic packaging, slightly cluttered shelves,
authentic household refrigerator.

No labels, no bounding boxes, no text overlays.
"""

image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=4.5,
    height=768,
    width=768
).images[0]

image.save("synthetic_fridge_001.png")

print("Image generated successfully")
