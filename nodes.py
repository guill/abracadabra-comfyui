import nodes
import os
import random
import re
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from .tools import VariantSupport
from comfy.graph_utils import GraphBuilder, is_link
from comfy.graph import DynamicPrompt

indentation_regex = re.compile(r"^[ \t]+")

ENABLE_ALL_NODES = False
def get_description(cls):
    docstr = cls.__doc__
    if docstr is None:
        return None
    return docstr.strip()

def get_available_nodes():
    if ENABLE_ALL_NODES:
        return nodes.NODE_CLASS_MAPPINGS.keys()
    else:
        return [
            "KSampler",
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "CLIPSetLastLayer",
            "VAEDecode",
            "VAEEncode",
            "VAEEncodeForInpaint",
            "VAELoader",
            "EmptyLatentImage",
            "LatentUpscale",
            "LatentUpscaleBy",
            "LatentFromBatch",
            "RepeatLatentBatch",
            "SaveImage",
            "PreviewImage",
            "LoadImage",
            "LoadImageMask",
            "ImageScale",
            "ImageScaleBy",
            "ImageInvert",
            "ImageBatch",
            "ImagePadForOutpaint",
            "EmptyImage",
            "ConditioningAverage",
            "ConditioningCombine",
            "ConditioningConcat",
            "ConditioningSetArea",
            "ConditioningSetAreaPercentage",
            "ConditioningSetAreaStrength",
            "ConditioningSetMask",
            "KSamplerAdvanced",
            "SetLatentNoiseMask",
            "LatentComposite",
            "LatentBlend",
            "LatentRotate",
            "LatentFlip",
            "LatentCrop",
            "LoraLoader",
            "CLIPLoader",
            "UNETLoader",
            "DualCLIPLoader",
            "CLIPVisionEncode",
            "StyleModelApply",
            "unCLIPConditioning",
            "ControlNetApply",
            "ControlNetApplyAdvanced",
            "ControlNetLoader",
            "DiffControlNetLoader",
            "StyleModelLoader",
            "CLIPVisionLoader",
            "VAEDecodeTiled",
            "VAEEncodeTiled",
            "unCLIPCheckpointLoader",
            "GLIGENLoader",
            "GLIGENTextBoxApply",
            "InpaintModelConditioning",
            "CheckpointLoader",
            "DiffusersLoader",
            "LoadLatent",
            "SaveLatent",
            "ConditioningZeroOut",
            "ConditioningSetTimestepRange",
            "LoraLoaderModelOnly",

            "Mask By Text",
            "Mask Morphology",
            "Combine Masks",
            "Unary Mask Op",
            "Unary Image Op",
            "Blur",
            "Image To Mask",
            "Mix Images By Mask",
            "Mix Color By Mask",
            "Mask To Region",
            "Cut By Mask",
            "Paste By Mask",
            "Get Image Size",
            "Change Channel Count",
            "Constant Mask",
            "Prune By Mask",
            "Separate Mask Components",
            "Create Rect Mask",
            "Make Image Batch",
            "Create QR Code",
            "Convert Color Space",
            "MasqueradeIncrementer",
        ]

def get_partial_graph_errors(graph: GraphBuilder, existing_graph: DynamicPrompt):
    errors = []
    for _, node in graph.nodes.items():
        class_type = node.class_type
        if class_type not in get_available_nodes():
            errors.append(f"Node type '{node['class_type']}' is not available in this environment.")
        cls = nodes.NODE_CLASS_MAPPINGS[class_type]
        inputs = cls.INPUT_TYPES()
        all_inputs = {}
        all_inputs.update(inputs.get("required", {}))
        all_inputs.update(inputs.get("optional", {}))
        for k, v in node.inputs.items():
            if k not in all_inputs:
                errors.append(f"Node type '{class_type}' does not have an input named '{k}'")
            else:
                input_type = all_inputs[k][0]
                if is_link(v):
                    from_id, idx = v
                    from_node = graph.nodes.get(from_id, None)
                    from_class_type = None
                    if from_node is None:
                        from_node = existing_graph.get_node(from_id)
                        if from_node is not None:
                            from_class_type = from_node["class_type"]
                    else:
                        from_class_type = from_node.class_type
                    if from_node is None:
                        errors.append(f"Node of type '{class_type}' has an input from an invalid node: '{from_id}'")
                        continue
                    assert from_class_type is not None
                    from_cls = nodes.NODE_CLASS_MAPPINGS[from_class_type]
                    from_outputs = from_cls.RETURN_TYPES
                    if idx >= len(from_outputs):
                        errors.append(f"Node of type '{class_type}' is attempting to use output {idx} from node of type '{from_class_type}' which only has {len(from_outputs)} outputs.")
                        continue
                    from_output = from_outputs[idx]
                    if from_output != input_type:
                        errors.append(f"The {k} input of node of type '{class_type}' is expecting type '{input_type}' but got type '{from_output}' from node of type '{from_class_type}'.")
                        continue
        for k, v in inputs.get("required", []).items():
            if k not in node.inputs:
                errors.append(f"Node of type '{class_type}' is missing a required input '{k}'.")
    if len(errors) > 0:
        return errors
    return None

def get_node_summaries():
    result = ""
    for nodename in get_available_nodes():
        nodedef = nodes.NODE_CLASS_MAPPINGS[nodename]
        result += f"Node: '{nodename}'\n"
        description = get_description(nodedef)
        if description:
            result += f"  Description: {description}\n"
        inputs = nodedef.INPUT_TYPES()
        required = inputs.get("required", {})
        if len(required) > 0:
            result += f"  Required Inputs:\n"
            for k, v in required.items():
                if isinstance(v[0], list):
                    result += f"    '{k}': OneOf{v[0]}\n"
                else:
                    result += f"    '{k}': {v[0]}\n"
        optional = inputs.get("optional", {})
        if len(optional) > 0:
            result += f"  Optional Inputs:\n"
            for k, v in optional.items():
                if isinstance(v[0], list):
                    result += f"    '{k}': OneOf{v[0]}\n"
                else:
                    result += f"    '{k}': {v[0]}\n"
        if len(required) == 0 and len(optional) == 0:
            result += f"  No Inputs\n"

        outputs = nodedef.RETURN_TYPES
        output_names = getattr(nodedef, "RETURN_NAMES", None)
        if len(outputs) > 0:
            result += f"  Outputs:\n"
            for idx in range(len(outputs)):
                v = outputs[idx]
                if output_names is not None and len(output_names) > idx:
                    result += f"    {idx}: {v} - '{output_names[idx]}'\n"
                else:
                    result += f"    {idx}: {v}\n"
        else:
            result += f"  No Outputs\n"
    return result

class AbracadabraNodeDefSummary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "summary"
    CATEGORY = "Abracadabra"

    def summary(self):
        return (get_node_summaries(),)

@VariantSupport()
class AbracadabraNode:
    def __init__(self):
        pass

    NUM_INPUTS = 5
    NUM_OUTPUTS = 5
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instructions": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                f"input{i}": ("*", {"rawLink": True}) for i in range(1, cls.NUM_INPUTS + 1)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
            }
        }

    RETURN_TYPES = tuple(["*"] * NUM_OUTPUTS)
    FUNCTION = "do_magic"

    CATEGORY = "Abracadabra"

    def do_magic(self, instructions, seed, dynprompt, **kwargs):
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are tasked with developing node-graph based workflows according to the user's instructions. You will be given the list of available nodes as well as a number of examples of creating workflows using those nodes. Your task is to respond with a chunk of Python code that creates a node graph to fulfill the user's request. You should not do ANY work in Python other than creating the node graphs. You should never use loops or conditionals in Python. Instead, make use of image batches when possible. (All IMAGE types are actually a batch of images.) Ensure you include all required inputs for each node."},
            {"role": "system", "content": "Here is the definition of available nodes. Do not attempt to use any nodes that are not listed here.\n\n" + get_node_summaries()},
        ]
        # Iterate through the 'examples/' folder and add the contents of each file to the messages
        examples_dir = os.path.join(os.path.dirname(__file__), "examples")
        test_code = ""
        match = "beaver"
        for filename in os.listdir(examples_dir):
            with open(os.path.join(examples_dir, filename), "r") as f:
                contents = f.read()
                prompt = ""
                first_line, rest = contents.split("\n", 1)
                while len(first_line) > 0 and first_line[0] == "#":
                    prompt += first_line[1:].strip() + "\n"
                    first_line, rest = rest.split("\n", 1)
                messages.append({"role": "user", "content": prompt})
                code = first_line + '\n' + rest
                messages.append({"role": "assistant", "content": code})
                if match in prompt:
                    test_code = code

        input_types = {}
        input_names = {}
        for k, v in kwargs.items():
            if is_link(v):
                from_id, idx = v
                class_type = dynprompt.get_node(from_id)['class_type']
                cls = nodes.NODE_CLASS_MAPPINGS[class_type]
                ret_type = cls.RETURN_TYPES[idx]
                if hasattr(cls, "RETURN_NAMES"):
                    input_names[k] = cls.RETURN_NAMES[idx]
                input_types[k] = ret_type
                print(f"Input {k} is a link from {from_id} with type {ret_type}")
            else:
                if isinstance(v, str):
                    input_types[k] = "STRING"
                elif isinstance(v, int):
                    input_types[k] = "INTEGER"
                elif isinstance(v, float):
                    input_types[k] = "FLOAT"
                elif isinstance(v, bool):
                    input_types[k] = "BOOLEAN"
                else:
                    input_types[k] = type(v).__name__
        prompt = f"""Instruction: {instructions}
Available locals:
- g: GraphBuilder
- RAND: fn() -> int
- result: dict - Set the 'outputs' key to a list of outputs to return. Ensure that it is a list and not a single value.
"""
        for k, v in input_types.items():
            if k in input_names:
                prompt += f"- {k}: {v} - Comes from output named '{input_names[k]}'. Pass directly to sockets of type {v} as {input_names[k]}\n"
            else:
                prompt += f"- {k}: {v} - Pass directly to sockets of type {v}\n"
        messages.append({"role": "user", "content": prompt})

        client = OpenAI()
        code = ""
        for _ in range(3):
            print("Requesting completion from OpenAI:\n\n", messages, "\n\n")
            completion = client.chat.completions.create(
                # model="gpt-4-turbo-preview",
                model="gpt-3.5-turbo",
                messages=messages
            )
            response = completion.choices[0].message
            code = response.content
            assert code is not None
            if "```" in code:
                code = code.split("```")[1]
            print("Got response from OpenAI:\n\n", code, "\n\n")
            # code = test_code
            if indentation_regex.match(code) is not None:
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "system", "content": f"You must not use any Python flow control (while loops, conditionals, etc.). All your work must be done via the node graph. Fix this issue and try again."})
                print("Error in generated graph (flow control). Trying again:\n")
                continue
            try:
                objcode = compile(code, "<string>", "exec")
                result = {}
                builder = GraphBuilder()
                generator = random.Random(seed)
                def rand():
                    return generator.randint(0, 0xffffffffffffffff)

                locals = {
                    "g": builder,
                    "RAND": rand,
                    "result": result,
                    **kwargs
                }
                globals = {}
                exec(objcode, globals, locals)
            except Exception as e:
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "system", "content": f"Your code failed with the following error. Please fix that error and try again. Do not apologize -- just respond with the updated code. Error: {e}"})
                print("Error in generated graph. Trying again:\n", e)
                continue
            errors = get_partial_graph_errors(builder, dynprompt)
            if errors is not None:
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "system", "content": f"Your code failed to generate a valid graph. Please fix the following errors and try again. Do not apologize -- just respond with the updated code. Errors: {errors}"})
                print("Error in generated graph. Trying again:\n", errors)
                continue
            return {
                "result": tuple(result['outputs']),
                "expand": builder.finalize(),
            }

        raise Exception(f"Failed to generate a valid response: {code}")

NODE_CLASS_MAPPINGS = {
    "AbracadabraNodeDefSummary": AbracadabraNodeDefSummary,
    "AbracadabraNode": AbracadabraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AbracadabraNodeDefSummary": "Abracadabra Summary",
    "AbracadabraNode": "Abracadabra",
}
