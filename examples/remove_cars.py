# Instruction: Photoshop all cars out of the image
# Available locals:
# - g: GraphBuilder
# - RAND: fn() -> int
# - result: dict - Set the 'outputs' key to a list of outputs to return
# - input1: IMAGE - Pass directly to sockets of type IMAGE

# The faces likely look bad because they were generated at low resolution. We'll try to fix them by cutting them out, inpainting at a higher resolution, and then pasting them back into the original image.
car_mask_raw = g.node("Mask By Text", image=input1, prompt="car", negative_prompt="", precision=0.3, normalize="no")
# Make sure we cut out the entire face
car_open = g.node("Mask Morphology", image=car_mask_raw.out(0), op="open", distance=10)
car_dilated = g.node("Mask Morphology", image=car_open.out(0), op="dilate", distance=8)
# Inpainting requires an input of type "MASK", so we'll use Image to Mask to convert the image to a mask.
inpaint_mask = g.node("Image To Mask", image=car_dilated.out(0), method="intensity")
loader = g.node("CheckpointLoaderSimple", ckpt_name="sd-v1-5-inpainting.ckpt")
vae_encode = g.node("VAEEncodeForInpaint", pixels=input1, mask=inpaint_mask.out(0), vae=loader.out(2), grow_mask_by=6)
prompt = g.node("CLIPTextEncode", clip=loader.out(1), text="")
negative = g.node("CLIPTextEncode", clip=loader.out(1), text="car")
# Use 80 steps for better inpainting
sampler = g.node("KSampler", model=loader.out(0), positive=prompt.out(0), negative=negative.out(0), latent_image=vae_encode.out(0), seed=RAND(), steps=80, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=0.9,)
decoded = g.node("VAEDecode", samples=sampler.out(0), vae=loader.out(2))
result['outputs'] = [decoded.out(0)]
