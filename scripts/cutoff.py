from collections import defaultdict
from typing import Union, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import gradio as gr

import re
import time

from modules.processing import StableDiffusionProcessing
from modules import scripts

from scripts.cutofflib.sdhook import SDHook
from scripts.cutofflib.embedding import CLIP, generate_prompts, token_to_block
from scripts.cutofflib.utils import log, set_debug
from scripts.cutofflib.xyz import init_xyz

NAME = 'Cutoff'
PAD = '_</w>'

def check_neg(s: str, negative_prompt: str, all_negative_prompts: Union[List[str],None]):
    if s == negative_prompt:
        return True
    
    if all_negative_prompts is not None:
        return s in all_negative_prompts
    
    return False

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # cf. https://memo.sugyan.com/entry/2022/09/09/230645

    inputs_are_torch = False
    input_device = v0.device
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


class Hook(SDHook):
    
    def __init__(
        self,
        enabled: bool,
        targets: List[str],
        padding: Union[str,int],
        weight: float,
        disable_neg: bool,
        strong: bool,
        interpolate: str,
    ):
        super().__init__(enabled)
        self.targets = targets
        self.padding = padding
        self.weight = float(weight)
        self.disable_neg = disable_neg
        self.strong = strong
        self.intp = interpolate
    
    def interpolate(self, t1: Tensor, t2: Tensor, w):
        if self.intp == 'lerp':
            return torch.lerp(t1, t2, w)
        else:
            return slerp(w, t1, t2)
    
    def hook_clip(self, p: StableDiffusionProcessing, clip: nn.Module):
        
        skip = False
        
        def hook(clip: nn.Module, inputs: Tuple[List[str]], output: Tensor):
            nonlocal skip
            
            if skip:
                # called from <A> below
                return
            
            assert isinstance(clip, CLIP)
            
            prompts, *rest = inputs
            assert len(prompts) == output.shape[0]
            
            # Check whether we are processing Negative prompt or not.
            # I firmly believe there is no one who uses a negative prompt 
            # exactly identical to a prompt.
            if self.disable_neg:
                if all(check_neg(x, p.negative_prompt, p.all_negative_prompts) for x in prompts):
                    # Now we are processing Negative prompt and skip it.
                    return
            
            output = output.clone()
            for pidx, prompt in enumerate(prompts):
                prompt_tokens = token_to_block(clip, prompt)
                
                cutoff = generate_prompts(clip, prompt, self.targets, self.padding)
                switch_base = np.full_like(cutoff.sw, self.strong)
                switch = np.full_like(cutoff.sw, True)
                active = cutoff.active_blocks()
                
                prompt_to_tokens = defaultdict(lambda: [])
                for token_idx, (token, block_index) in enumerate(prompt_tokens):
                    if block_index in active:
                        sw = switch.copy()
                        sw[block_index] = False
                        prompt = cutoff.text(sw)
                    else:
                        prompt = cutoff.text(switch_base)
                    prompt_to_tokens[prompt].append((token_idx, token))
                
                #log(prompt_to_tokens)
                
                token_keys = list(prompt_to_tokens.keys())
                if len(token_keys) == 0:
                    # without any (negative) prompts
                    token_keys.append('')
                
                try:
                    # <A>
                    skip = True
                    vs = clip(token_keys)
                finally:
                    skip = False
                
                tensor = output[pidx, :, :] # e.g. (77, 768)
                for token_key, t in zip(token_keys, vs):
                    if tensor.shape == t.shape:
                        #assert tensor.shape == t.shape
                        for token_idx, token in prompt_to_tokens[token_key]:
#                            log(f'{token_idx:03} {token.token:<16} {token_key}')
                            tensor[token_idx, :] = self.interpolate(tensor[token_idx,:], t[token_idx,:], self.weight)
                    else:
                        log("Webui-cutoff-fork: Warning: tensor shape != t.shape, something is weird. Skipping iteration.")

            return output
        
        self.hook_layer(clip, hook)
    

def _get_effective_prompt(prompts: list[str], prompt: str) -> str:
    return prompts[0] if prompts else prompt


class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
        self.last_hooker: Union[SDHook,None] = None

    def title(self):
        return NAME
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def stripSugarFromAllPrompts(self, p):
        if (p.all_prompts):
            for i, prompt in enumerate(p.all_prompts):
                p.all_prompts[i] = p.all_prompts[i].replace("&&", '')

    # Extract tokens, remove sugar (&&foo&& -> foo)
    def getTargetsFromPromptAndStripDelimiters(self, p):
        original_prompt = _get_effective_prompt(p.all_prompts, p.prompt)
        if "&&" in original_prompt:
            targets = []
            sections = re.split('(&&.*?&&)', p.prompt, flags=re.DOTALL)
            for i, section in enumerate(sections):
                if section.startswith('&&') and section.endswith('&&'):
                    # This is a target word. Grab it.
                    target = section[2:-2].strip()  # Remove the delimiters
                    targets.append(target)
                    # Effectively strips the delimiter from the prompt
                    sections[i] = target
            # Join the sections back together into the final prompt
            prompt_out = ''.join(sections)
            p.prompt = prompt_out
            self.stripSugarFromAllPrompts(p)
            return targets
        else:
            return []


    def ui(self, is_img2img):
        with gr.Accordion(NAME + " in Prompt", open=False):
            enabled = gr.Checkbox(label='Enabled', value=False)

            targets = gr.Textbox(label='Target tokens (comma separated)', placeholder='red, blue')
            weight = gr.Slider(minimum=-1.0, maximum=2.0, step=0.01, value=0.5, label='Weight')
            with gr.Accordion('Details', open=False):
                disable_neg = gr.Checkbox(value=True, label='Disable for Negative prompt.')
                embedded_targets_disabled = gr.Checkbox(label='Disable &&--&& syntax in main prompt', value=False)
                default_targets_disabled = gr.Checkbox(label='Do not include default targets', value=False)
                strong = gr.Checkbox(value=False, label='Cutoff strongly.')
                padding = gr.Textbox(label='Padding token (ID or single token)')
                lerp = gr.Radio(choices=['Lerp', 'SLerp'], value='Lerp', label='Interpolation method')
            
            debug = gr.Checkbox(value=False, label='Debug log')
            debug.change(fn=set_debug, inputs=[debug], outputs=[])
                
        return [
            enabled,
            embedded_targets_disabled,
            default_targets_disabled,
            targets,
            weight,
            disable_neg,
            strong,
            padding,
            lerp,
            debug,
        ]
    
    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        embedded_targets_disabled: bool,
        default_targets_disabled: bool,
        targets_: str,
        weight: Union[float,int],
        disable_neg: bool,
        strong: bool,
        padding: Union[str,int],
        intp: str,
        debug: bool,
    ):
        set_debug(debug)
        
        if self.last_hooker is not None:
            self.last_hooker.__exit__(None, None, None)
            self.last_hooker = None
        
        if not enabled:
            return
        
        unique_colors = [
            "Red", "Blue", "Yellow", "Green", "Black", "White", "Brown", "Orange",
            "Purple", "Pink", "Gray", "Violet", "Maroon", "Gold", "Silver", "Beige",
            "Cyan", "Magenta", "Turquoise", "Tan", "Olive", "Indigo", "Charcoal",
            "Navy", "Teal", "Lime", "Lavender", "Peach", "Emerald", "Ruby", "Salmon",
            "Plum", "Coral", "Fuchsia", "Amber", "Azure", "Rose", "Jade", "Lemon",
            "Cream", "Pearl", "Chocolate", "Ivory", "Champagne", "Slate", "Mustard",
            "Raspberry", "Burgundy", "Eggplant", "Aquamarine", "Crimson", "Imperial Yellow",
            "Chartreuse", "Marigold", "Amethyst", "Lilac", "Garnet", "Topaz", "Periwinkle",
            "Cobalt", "Orchid", "Citrine", "Vermilion", "Pewter", "Sienna", "Sapphire",
            "Bronze", "Turmeric", "Steel", "Onyx", "Sand", "Mulberry", "Carnation",
            "Jadeite", "Paprika", "Hibiscus", "Citron", "Tangerine", "Honeydew", "Caramel",
            "Pomegranate", "Cinnamon", "Fern", "Butterscotch", "Petal", "Ochre", "Pistachio",
            "Papaya", "Platinum", "Carnelian", "Eucalyptus", "Moonstone", "Mauve"
        ]

        #log("Webui-cutoff-fork: Searching for color words in prompts.")
        start_time = time.time()

        found_color_words = []
        if (not default_targets_disabled):
            for color in unique_colors:
                color_pattern = re.compile(re.escape(color), re.IGNORECASE | re.DOTALL)
                for prompt in p.all_prompts:
                    matches = color_pattern.findall(prompt)
                    for matchingColorWord in matches:
                        found_color_words.append(matchingColorWord)

        end_time = time.time()

        elapsed_time = end_time - start_time
        log(f"Webui-cutoff-fork: Default target search time taken: {elapsed_time:.6f} seconds")

        found_color_words = list(set(found_color_words))

        if (not embedded_targets_disabled):
            targets = self.getTargetsFromPromptAndStripDelimiters(p)
        else:
            if targets_ is None or len(targets_) == 0:
                targets = []
            else:
                targets = [x.strip() for x in targets_.split(',')]
                targets = [x for x in targets if len(x) != 0]

        for color in found_color_words:
            if color in targets:
                targets.remove(color)

        targets = targets + found_color_words

        if len(targets) == 0:
            return
        
        if padding is None:
            padding = PAD
        elif isinstance(padding, str):
            if len(padding) == 0:
                padding = PAD
            else:
                try:
                    padding = int(padding)
                except:
                    if not padding.endswith('</w>'):
                        padding += '</w>'
        
        weight = float(weight)
        intp = intp.lower()
        
        self.last_hooker = Hook(
            enabled=True,
            targets=targets,
            padding=padding,
            weight=weight,
            disable_neg=disable_neg,
            strong=strong,
            interpolate=intp,
        )
        
        self.last_hooker.setup(p)
        self.last_hooker.__enter__()
        
        p.extra_generation_params.update({
            f'{NAME} enabled': enabled,
            f'{NAME} targets': targets,
            f'{NAME} padding': padding,
            f'{NAME} weight': weight,
            f'{NAME} disable_for_neg': disable_neg,
            f'{NAME} strong': strong,
            f'{NAME} interpolation': intp,
        })

init_xyz(Script, NAME)
