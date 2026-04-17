import os
import cv2
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
import torch
from segmenter_gui import ROOT, PROJECT_DIR
import yaml
import time
import numpy as np


def load_config():
    with open(os.path.join(ROOT, 'config.yaml'), 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

def test_torch():
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    assert torch.cuda.is_available()
    print('Number of devices:', torch.cuda.device_count())
    assert torch.cuda.device_count() > 0
    print('Devices:')
    for device in range(torch.cuda.device_count()):
        print(f'{device}:', torch.cuda.get_device_name(device))

class image_processing():
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.org_image = None
        self.filename = None
        self.alpha = 1
        self.beta = 0.65
        self.preview_scale=0.7


    def load_image(self, image_path:str, tobe_resized:bool=False) :
        image =cv2.imread(image_path)
        self.org_image = image.copy()
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if tobe_resized:
            image = image.resize(
                (self.CONFIG["sam3"]["default_size"], self.CONFIG["sam3"]["default_size"])
            )
        return image

    def merge_all_mask(self, nchw_array:np.ndarray) -> np.ndarray:
        nhw_array = np.squeeze(nchw_array, axis=1)
        hw_array= np.any(nhw_array, axis=0).astype(bool)
        return hw_array

    def visualize_mask(self, mask:np.ndarray, is_gui:bool=False):
        anns_color = np.random.randint(0, 255, size=3)
        background = np.ones_like(self.org_image)
        merge_mask = self.merge_all_mask(mask)
        background[merge_mask] = anns_color

        if is_gui:
            vis = cv2.addWeighted(self.org_image,
                                  self.alpha,
                                  background,
                                  self.beta, 0)

            vis = cv2.resize(vis, None,
                             fx=self.preview_scale,
                             fy=self.preview_scale,
                             interpolation=cv2.INTER_LANCZOS4)

            cv2.imshow("vis", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()





class segmenter_model:
    def __init__(self, CONFIG, model_name:str):
        self.CONFIG = CONFIG
        self.model_name = model_name

    def build_model(self):
        if self.model_name == "sam3":
            print("building sam3 model")

            try:
                start = time.time()

                model = build_sam3_image_model(
                        bpe_path=self.CONFIG["sam3"]["bpe_path"],
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        eval_mode=self.CONFIG["sam3"]["eval_mode"],
                        checkpoint_path=self.CONFIG["sam3"]["checkpoint_path"],
                        load_from_HF=self.CONFIG["sam3"]["load_from_HF"],
                        enable_inst_interactivity=self.CONFIG["sam3"]["enable_inst_interactivity"],
                    ).float()

                model = Sam3Processor(model, confidence_threshold=self.CONFIG["sam3"]["conf_thd"])

                end = time.time()
                print("successfully built sam3 model in {:.2f}s".format(end - start))

            except Exception as e:
                print(e)
                exit()
        else:
            print("model name unknown!!")
            exit()

        return model

    def image_inference(self,
                        image:np.ndarray,
                        text_prompt:str,
                        segmenter,
                        to_cpu:bool=False):
        inference_state = segmenter.set_image(image)
        start = time.time()
        with torch.no_grad():
            output = segmenter.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )
        end = time.time()
        print("")
        print(f"-------- Inference finished in {(end - start) * 1000:.2f} ms")
        print(f"prompt '{text_prompt}' found {len(output["scores"])} object(s)")
        print(f"")
        if to_cpu:
            return output["masks"].cpu().numpy(), output["boxes"].cpu().numpy(), output["scores"].cpu().numpy()
        else:
            return output["masks"], output["boxes"], output["scores"]




