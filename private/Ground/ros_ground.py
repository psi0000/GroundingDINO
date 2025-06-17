import rospy
import json
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from groundingdino.util.inference import load_model, annotate, predict
from groundingdino.datasets import transforms as DINOTransforms
from PIL import Image as PILImage
import torch
import difflib

def groundingdino_preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    transform = DINOTransforms.Compose([
        DINOTransforms.RandomResize([800], max_size=1333),
        DINOTransforms.ToTensor(),
        DINOTransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PILImage.fromarray(image_rgb)
    image_transformed, _ = transform(image_pil, None)
    return image_bgr, image_transformed

def clean_phrase(phrase: str) -> str:
    words = phrase.strip().split()
    seen = set()
    return ' '.join([w for w in words if not (w in seen or seen.add(w))])

def match_phrase_to_label(phrase, label_list):
    best_match = difflib.get_close_matches(phrase, label_list, n=1, cutoff=0.5)
    return best_match[0] if best_match else phrase

class GroundingDINORosNode:
    def __init__(self):
        rospy.init_node('groundingdino_node')

        cfg_path = "/root/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights_path = "/root/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.model = load_model(cfg_path, weights_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.box_thresh = 0.35
        self.text_thresh = 0.35

        self.bridge = CvBridge()
        self.latest_img = None
        self.latest_prompt = None
        self.default_prompt = "fire"

        rospy.Subscriber('/rgb_image', Image, self._image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/ollama_msg', String, self._prompt_cb, queue_size=1)
        self.box_pub = rospy.Publisher('/groundingdino/boxes', String, queue_size=1)
        self.img_pub = rospy.Publisher('/groundingdino/annotated_image', Image, queue_size=1)

        rospy.wait_for_message('/rgb_image', Image)
        self.rate = rospy.Rate(10)

    def _image_cb(self, msg: Image):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            self.latest_img = None

    def _prompt_cb(self, msg: String):
        try:
            parsed = json.loads(msg.data)
            if "danger_zones" in parsed:
                labels = [item["label"] for item in parsed["danger_zones"] if "label" in item]
                self.latest_prompt = " . ".join(labels)
            elif isinstance(parsed, list) and all("label" in item for item in parsed):
                labels = [item["label"] for item in parsed]
                self.latest_prompt = " . ".join(labels)
            else:
                self.latest_prompt = msg.data
        except Exception:
            self.latest_prompt = msg.data
        rospy.loginfo(f"[Ollama] Prompt updated: {self.latest_prompt}")

    def spin(self):
        while not rospy.is_shutdown():
            if self.latest_img is not None:
                try:
                    prompt_to_use = self.latest_prompt if self.latest_prompt else self.default_prompt

                    label_list = [label.strip() for label in prompt_to_use.split('.') if label.strip()]

                    image_source, image = groundingdino_preprocess(self.latest_img)
                    image = image.to(self.device)

                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=image,
                        caption=prompt_to_use,
                        box_threshold=self.box_thresh,
                        text_threshold=self.text_thresh
                    )

                    clean_phrases = [clean_phrase(p) for p in phrases]

                    results = [
                        {
                            'box': b.tolist(),
                            'score': float(s),
                            'phrase': p,
                            'matched_label': match_phrase_to_label(p, label_list)
                        }
                        for b, s, p in zip(boxes, logits, clean_phrases)
                    ]

                    self.box_pub.publish(String(json.dumps(results)))

                    annotated = annotate(
                        image_source=image_source,
                        boxes=boxes,
                        logits=logits,
                        phrases=clean_phrases
                    )
                    cv2.imwrite("/root/vlm_ws/src/GroundingDINO/private/Ground/ros_test.jpg", annotated)
                    out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                    out_msg.header.stamp = rospy.Time.now()
                    out_msg.header.frame_id = 'camera'
                    self.img_pub.publish(out_msg)

                except Exception as e:
                    rospy.logerr(f"[Inference error] {e}")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = GroundingDINORosNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass