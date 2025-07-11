from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("/root/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/root/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "image.png"
TEXT_PROMPT = "top region of Fire extinguisher . person . door . Hanging objects ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
print("Boxes:", boxes)
print("Boxes.shape:", boxes.shape)
cv2.imwrite("annotated_image.jpg", annotated_frame)