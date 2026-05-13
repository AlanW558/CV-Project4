import os
import sys
import torch
from diffusers.utils import load_image

torch.backends.cudnn.enabled = False

DIFIX_SRC = "/data3/yywong558/part3/Difix3D/src"
if DIFIX_SRC not in sys.path:
    sys.path.append(DIFIX_SRC)

from pipeline_difix import DifixPipeline


class DifixRefiner:
    def __init__(
        self,
        model_dir="/data3/yywong558/part3/difix",
        device=None,
        height=512,
        width=512,
        prompt="remove degradation",
        timestep=199,
    ):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.height = height
        self.width = width
        self.prompt = prompt
        self.timestep = timestep

        self.pipe = DifixPipeline.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def refine(self, input_image_path, ref_image_path, output_image_path):
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        image = load_image(input_image_path).convert("RGB").resize(
            (self.width, self.height)
        )
        ref_image = load_image(ref_image_path).convert("RGB").resize(
            (self.width, self.height)
        )

        output = self.pipe(
            prompt=self.prompt,
            image=image,
            ref_image=ref_image,
            height=self.height,
            width=self.width,
            num_inference_steps=1,
            timesteps=[self.timestep],
            guidance_scale=0.0,
        ).images[0]

        output.save(output_image_path)
        return output_image_path