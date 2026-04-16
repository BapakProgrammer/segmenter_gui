import warnings
warnings.filterwarnings("ignore")

import torch
from safetensors.torch import load_file
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import time

start_time = time.time()
model = build_sam3_image_model(
                           bpe_path= "./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                           device= "cuda" if torch.cuda.is_available() else "cpu",
                           eval_mode = True,
                           checkpoint_path = "./assets/sam3.pt",
                           load_from_HF = False,
                           enable_inst_interactivity = False,
                           )
end_time = time.time()
print(f'Time taken: {end_time - start_time} s')

processor = Sam3Processor(model)
