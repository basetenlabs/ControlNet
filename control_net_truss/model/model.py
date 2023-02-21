from typing import Dict, List

from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._ddim_sampler = None

    def load(self):
        self._model = create_model('./models/cldm_v15.yaml').cpu()
        self._model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
        self._model = self._model.cuda()
        self._ddim_sampler = DDIMSampler(self._model)



    def predict(self, request: Dict) -> Dict[str, List]:
        input_image = request.get("input_image", None) # type="numpy"
        prompt : str = request.get("prompt", None)
        a_prompt :str = request.get("a_prompt", None)
        n_prompt :str = request.get("n_prompt", None)
        num_samples = request.get("num_samples", 1)
        image_resolution = request.get("image_resolution", 512)
        ddim_steps = request.get("ddim_steps", 20)
        guess_mode = request.get("guess_mode", False)
        strength = request.get("strength", 1.0)
        scale = request.get("scale", 9.0)
        seed = request.get("seed", random.randint(1, 2147483647))
        eta = request.get("eta", 0.0)
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = np.zeros_like(img, dtype=np.uint8)
            detected_map[np.min(img, axis=2) < 127] = 255

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self._model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self._model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self._model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self._model.low_vram_shift(is_diffusing=True)

            self._model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self._ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self._model.low_vram_shift(is_diffusing=False)

            x_samples = self._model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return [255 - detected_map] + results

