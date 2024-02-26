# Instruction: Generate two images of cats on skateboards. Avoid having humans in the images.
# Available locals:
# - g: GraphBuilder
# - RAND: fn() -> int
# - result: dict - Set the 'outputs' key to a list of outputs to return
# - input1: MODEL - Pass directly to sockets of type MODEL
# - input2: CLIP - Pass directly to sockets of type CLIP
# - input3: VAE - Pass directly to sockets of type VAE
prompt = g.node("CLIPTextEncode", clip=input2, text="a cat on a skateboard")
negative = g.node("CLIPTextEncode", clip=input2, text="human, person")
empty_latent = g.node("EmptyLatentImage", width=512, height=512, batch_size=2)
sampler = g.node(
    "KSampler",
    model=input1,
    positive=prompt.out(0),
    negative=negative.out(0),
    latent_image=empty_latent.out(0),
    seed=RAND(),
    steps=30,
    cfg=8.0,
    sampler_name="dpmpp_2m",
    scheduler="karras",
    denoise=1.0,
)
decode = g.node("VAEDecode", samples=sampler.out(0), vae=input3)
result['outputs'] = [decode.out(0)]
