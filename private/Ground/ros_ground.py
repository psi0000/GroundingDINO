#!/usr/bin/env python3
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

def groundingdino_preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    
    transform = DINOTransforms.Compose([
        DINOTransforms.RandomResize([800], max_size=1333),
        DINOTransforms.ToTensor(),
        DINOTransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    # BGR → RGB → PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PILImage.fromarray(image_rgb)
    image_transformed, _ = transform(image_pil, None)  # ← numpy X, PIL O
    return image_bgr, image_transformed


class GroundingDINORosNode:
    def __init__(self):
        rospy.init_node('groundingdino_node')
        # 모델 로딩
        cfg_path = "/root/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights_path = "/root/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.model = load_model(cfg_path, weights_path)
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        # 임계값 설정
        self.box_thresh = 0.35
        self.text_thresh = 0.25
        # ROS 설정
        self.bridge = CvBridge()
        self.latest_img = None
        self.latest_prompt = "person . fire extinguisher . door . hanging object ."
        rospy.Subscriber('/rgb_image', Image, self._image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/ollama_msg', String, self._prompt_cb, queue_size=1)
        self.box_pub = rospy.Publisher('/groundingdino/boxes', String, queue_size=1)
        self.img_pub = rospy.Publisher('/groundingdino/annotated_image', Image, queue_size=1)
        rospy.wait_for_message('/rgb_image', Image)
        self.rate = rospy.Rate(10)

    def _prompt_cb(self, msg: String):
        self.latest_prompt = msg.data

    def _image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_img = img
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            self.latest_img = None

    def spin(self):
        while not rospy.is_shutdown():
            if self.latest_img is not None and self.latest_prompt:
                try:
                    # Preprocess using GroundingDINO's transforms
                    image_source, image = groundingdino_preprocess(self.latest_img)
                    image = image.to(self.device)
                    # 모델 추론
                    print(f"[DEBUG] predict used: {predict.__module__} | {predict.__name__}")

                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=image,
                        caption=self.latest_prompt,
                        box_threshold=self.box_thresh,
                        text_threshold=self.text_thresh
                    )
                    # 결과 발행
                    results = [
                        {'box': b.tolist(), 'score': float(s), 'phrase': p}
                        for b, s, p in zip(boxes, logits, phrases)
                    ]
                    print(f"[DEBUG] predict returned type: {type(results)}, value: {results}")
                    self.box_pub.publish(String(json.dumps(results)))
                    # 어노테이션 생성
                    annotated = annotate(
                        image_source=image_source,
                        boxes=boxes,
                        logits=logits,
                        phrases=phrases
                    )
                    # cv2.imwrite("/root/vlm_ws/src/GroundingDINO/private/Ground/ros_test.jpg", annotated)
                    out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                    out_msg.header.stamp = rospy.Time.now()
                    out_msg.header.frame_id = 'camera'
                    self.img_pub.publish(out_msg)
                except Exception as e:
                    print(f"[Inference error] {e}")
            self.rate.sleep()


if __name__ == '__main__':
    try:
        node = GroundingDINORosNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
