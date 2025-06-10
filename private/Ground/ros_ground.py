#!/usr/bin/env python3
import rospy
import json
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from groundingdino.util.inference import load_model, annotate
from groundingdino.util.inference import predict as dino_predict
import torchvision.transforms as T
from PIL import Image as PILImage
import numpy as np
import torch
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
        # 파라미터 설정
        self.box_thresh = 0.35
        self.text_thresh = 0.25
        # 이미지 변환기
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
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
                    # numpy BGR → PIL RGB
                    rgb = cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2RGB)
                    pil_image = PILImage.fromarray(rgb)
                    # 이미지 전처리 (ToTensor → Normalize → Unsqueeze → to(device))
                    image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    # GroundingDINO 추론
                    boxes, logits, phrases = dino_predict(
                        model=self.model,
                        image=image_tensor,
                        caption=self.latest_prompt,
                        box_threshold=self.box_thresh,
                        text_threshold=self.text_thresh
                    )
                    results = [
                        {'box': b.tolist(), 'score': float(s), 'phrase': p}
                        for b, s, p in zip(boxes, logits, phrases)
                    ]
                    self.box_pub.publish(String(json.dumps(results)))
                    # 어노테이션 이미지 생성
                    annotated = annotate(
                        image_source=self.latest_img.copy(),
                        boxes=boxes,
                        logits=logits,
                        phrases=phrases
                    )
                    cv2.imwrite("/root/vlm_ws/src/GroundingDINO/private/Ground/ros_test.jpg", annotated)
                    out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                    out_msg.header.stamp = rospy.Time.now()
                    out_msg.header.frame_id = 'camera'
                    self.img_pub.publish(out_msg)
                except Exception as e:
                    rospy.logerr(f"[Inference error] {e}")
                    rospy.loginfo(f"Image type: {type(self.latest_img)}")
                    if self.latest_img is not None:
                        rospy.loginfo(f"Image shape: {self.latest_img.shape}")
            self.rate.sleep()
if __name__ == '__main__':
    try:
        node = GroundingDINORosNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass











김형진 (Hyungjin Kim)에 메시지 보내기









