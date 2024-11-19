

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
image_path = r"D:\PycharmProjects\segment-anything\dog.jpg"
device = "cuda"
sam_checkpoint = r"D:\PycharmProjects\segment-anything\checkpoints\sam_vit_h_4b8939.pth"
model_type = "vit_h"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
cv2.imwrite("mask.jpg", masks)
print("Mask is ready")