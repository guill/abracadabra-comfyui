# Instruction: Filter the incoming images by season. Send the images that show summer to the first output and images that show fall to the second.
# Available locals:
# - g: GraphBuilder
# - RAND: fn() -> int
# - result: dict - Set the 'outputs' key to a list of outputs to return
# - input1: IMAGE - Pass directly to sockets of type IMAGE
summer_mask = g.node("Mask By Text", image=input1, prompt="summer", negative_prompt="", precision=0.5, normalize="no")
fall_mask = g.node("Mask By Text", image=input1, prompt="fall", negative_prompt="", precision=0.5, normalize="no")
summer_average = g.node("Unary Mask Op", image=summer_mask.out(1), op="average")
fall_average = g.node("Unary Mask Op", image=fall_mask.out(1), op="average")
comparison = g.node("Combine Masks", image1=summer_average.out(0), image2=fall_average.out(0), op="greater", clamp_result="yes", round_result="yes")
summer_images = g.node("Prune By Mask", image=input1, mask=comparison.out(0))
comparison_inverted = g.node("Unary Mask Op", image=comparison.out(0), op="invert")
fall_images = g.node("Prune By Mask", image=input1, mask=comparison_inverted.out(0))
result['outputs'] = [summer_images.out(0), fall_images.out(0)]
