# Instruction: Generate a 512x512 image of a cat playing a piano
# Available locals:
# - g: GraphBuilder
# - RAND: fn() -> int
# - result: dict - Set the 'outputs' key to a list of outputs to return
loader = g.node("CheckpointLoaderSimple", ckpt_name="epicrealism_pureEvolutionV5.safetensors")
prompt = g.node("CLIPTextEncode", clip=loader.out(1), text="a cat playing a piano")
negative = g.node("CLIPTextEncode", clip=loader.out(1), text="")
empty_latent = g.node("EmptyLatentImage", width=512, height=512, batch_size=1)
sampler = g.node(
    "KSampler",
    model=loader.out(0),
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
decode = g.node("VAEDecode", samples=sampler.out(0), vae=loader.out(2))
result['outputs'] = [decode.out(0)]
