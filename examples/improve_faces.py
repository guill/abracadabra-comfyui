# Instruction: Fix the faces of the soccer players in the given images.
# Available locals:
# - g: GraphBuilder
# - RAND: fn() -> int
# - result: dict - Set the 'outputs' key to a list of outputs to return
# - input1: IMAGE - Pass directly to sockets of type IMAGE

# The faces likely look bad because they were generated at low resolution. We'll try to fix them by cutting them out, inpainting at a higher resolution, and then pasting them back into the original image.
face_mask_raw = g.node("Mask By Text", image=input1, prompt="face", negative_prompt="", precision=0.3, normalize="no")
# Make sure we cut out the entire face
face_open = g.node("Mask Morphology", image=face_mask_raw.out(0), op="open", distance=10)
face_dilated = g.node("Mask Morphology", image=face_open.out(0), op="dilate", distance=8)
# Separate out each face
face_components = g.node("Separate Mask Components", mask=face_dilated.out(0))
# Cut out the regions for inpainting
regions = g.node("Mask To Region", mask=face_components.out(0), padding=64, constraints="keep_ratio", constraint_x=2, constraint_y=2, min_width=0, min_height=0, batch_behavior="match_ratio")
# Note: We MUST use mask_mapping_optional on the following line because the number of masks may not match the number of input images
cut_image = g.node("Cut By Mask", image=input1, mask=regions.out(0), mask_mapping_optional=face_components.out(1), force_resize_width=512, force_resize_height=512)
# We don't use mask_mapping_optional because obviously the number of masks matches the number of masks
cut_masks = g.node("Cut By Mask", image=face_components.out(0), mask=regions.out(0), force_resize_width=512, force_resize_height=512)
# Perform inpainting
inpaint_image = g.node("Change Channel Count", image=cut_image.out(0), kind="RGB")
# Inpainting requires an input of type "MASK", so we'll use Image to Mask to convert the image to a mask.
inpaint_mask = g.node("Image To Mask", image=cut_masks.out(0), method="intensity")
loader = g.node("CheckpointLoaderSimple", ckpt_name="sd-v1-5-inpainting.ckpt")
vae_encode = g.node("VAEEncodeForInpaint", pixels=inpaint_image.out(0), mask=inpaint_mask.out(0), vae=loader.out(2), grow_mask_by=6)
prompt = g.node("CLIPTextEncode", clip=loader.out(1), text="photo of a smiling person")
negative = g.node("CLIPTextEncode", clip=loader.out(1), text="blurry, low resolution, mutated")
# Use 80 steps for better inpainting
sampler = g.node("KSampler", model=loader.out(0), positive=prompt.out(0), negative=negative.out(0), latent_image=vae_encode.out(0), seed=RAND(), steps=80, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=0.9,)
decoded = g.node("VAEDecode", samples=sampler.out(0), vae=loader.out(2))
# Paste the inpainted faces back into the original image
# We multiply the alpha by the mask to avoid pasting a hard edge
image_to_paste = g.node("Combine Masks", image1=decoded.out(0), image2=cut_masks.out(0), op="multiply_alpha", clamp_result="yes", round_result="no")
pasted = g.node("Paste By Mask", image_base=input1, image_to_paste=image_to_paste.out(0), mask=regions.out(0), mask_mapping_optional=face_components.out(1), resize_behavior="resize")
result['outputs'] = [pasted.out(0)]
