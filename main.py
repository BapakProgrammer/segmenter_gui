import warnings
warnings.filterwarnings("ignore")
from segmenter_gui.src.utils import load_config
from segmenter_gui.src.utils import segmenter_model, image_processing
import time

CONFIG = load_config()
SEGMENTER = segmenter_model(CONFIG, model_name="sam3")
segmenter_model = SEGMENTER.build_model()

image_processor = image_processing(CONFIG)

image = image_processor.load_image(
    image_path=CONFIG["images"]["dummy_path"],
    tobe_resized=False,
)
print("im_array ", image.shape)
masks, boxes, scores = SEGMENTER.image_inference(
    image=image, text_prompt="road", segmenter=segmenter_model, to_cpu=True
)

print(masks.shape)
