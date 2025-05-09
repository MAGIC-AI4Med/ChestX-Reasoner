import torch
from rich import print
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

from . import BaseLM


class CheXagent8B(BaseLM):
    def __init__(self, name=f"CheXagent", device="cuda", dtype=torch.bfloat16):
        super().__init__(name=name, device=device, dtype=dtype)
        cache_dir="/mnt/lustre/fanziqing/radfm/checkpoint"
        checkpoint = "StanfordAIMI/CheXagent-8b"
        print(f"=> Loading CheXagent3 model ({checkpoint})")
        self.checkpoint = checkpoint
        self.processor = AutoProcessor.from_pretrained(self.checkpoint,trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(self.checkpoint,cache_dir=cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, device_map=device, trust_remote_code=True,cache_dir=cache_dir
            ,local_files_only=True
        )
        self.model = self.model.to(self.dtype)
        self.model.eval()

    def get_prompt(self, question, options):
        choice_style = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        option_str = ", ".join([f"({choice_style[option_idx]}) {option}" for option_idx, option in enumerate(options)])
        option_notations = ", ".join([f"({choice_style[option_idx]})" for option_idx, option in enumerate(options)])
        prompt = f'{question}\nOptions: {option_str}\n' \
                 f'Directly answer the question by choosing one option: {option_notations}.'
        return prompt

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        return paths

    def generate(self, paths, prompt, **kwargs):
        paths = self.process_img(paths)
        images = [Image.open(path).convert("RGB") for path in paths]

        inputs = self.processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device="cuda", dtype=torch.float16)
        output = self.model.generate(**inputs, generation_config=self.generation_config)[0]
        response = self.processor.tokenizer.decode(output, skip_special_tokens=True)

        return response


if __name__ == '__main__':
    model = CheXagent()
