#!/usr/bin/env python3
import rospy
import json
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import base64
import requests
import re

class OllamaVQANode:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('ollama_vqa_node')
        # CvBridge 초기화
        self.bridge = CvBridge()
        self.latest_img = None
        # ROS 설정
        rospy.Subscriber('/rgb_image', Image, self._image_cb, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher('/ollama_msg', String, queue_size=1)
        # Ollama 모델 이름
        self.model_name = "qwen2.5vl"
        # 프롬프트 정의
        self.prompt = """
        드론의 실내 비행 중 촬영한 온보드 카메라 이미지가 제공됩니다.

        당신의 임무는 드론이 피해야 할 특정 구역을 식별하는 것입니다. 이는 물리적 장애물이 아니라도 다음과 같은 이유로 위험할 수 있습니다:

        - 중요한 안전 기능 (예: 경보장치, 소화기)
        - 예측 불가능한 동작 (예: 열린 문, 매달린 물체)
        - RGB 센서가 잘못 해석할 수 있는 의미적으로 모호한 영역 (예: 반사, 유리, 그림자)

        각 위험 구역에 대해:
        - 객체 또는 구역을 설명하는 짧은 라벨 제공
        - 왜 위험한지 설명
        - 드론의 현재 시점에서 위험 구역을 기준으로 안전한 이동 방향을 제안. 다음 표현 중 하나 사용: "move left", "move right", "move up", "move down"

        추가로:
        - 처리에 사용된 내부 이미지 해상도. 크기 조정되지 않았다면 "original" 반환

        다음 엄격한 JSON 형식으로만 영어로 응답:

        {
        "internal_image_resolution": "original",
        "danger_zones": [
            {
            "label": "구역의 간결한 설명",
            "reason": "위험한 이유",
            "safe_movement_direction": "move left | move right | move up | move down",
            "risk region position": {
                "bbox_2d": [x_min, y_min, x_max, y_max]
            }
            }
        ]
        }
        """
        # ROS 메시지 대기
        rospy.wait_for_message('/rgb_image', Image)
        self.rate = rospy.Rate(10)  # 10Hz로 처리

    def _image_cb(self, msg: Image):
        try:
            # ROS Image 메시지를 OpenCV BGR 이미지로 변환
            self.latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 에러: {e}")
            self.latest_img = None

    def encode_image_to_base64(self, image: np.ndarray):
        # OpenCV 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode("utf-8")

    def run_ollama_vqa_and_extract_info(self, image: np.ndarray):
        # 이미지 Base64로 인코딩
        image_b64 = self.encode_image_to_base64(image)
        
        # Ollama API 요청 페이로드
        payload = {
            "model": self.model_name,
            "prompt": self.prompt,
            "images": [image_b64],
            "stream": True
        }

        try:
            # Ollama API로 요청
            response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
            response.raise_for_status()

            # 스트림 응답 조립
            full_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        part = json.loads(line.decode("utf-8"))
                        full_text += part.get("response", "")
                    except Exception:
                        continue

            full_text = full_text.strip("`\n ")
            if full_text.lower().startswith("json"):
                full_text = full_text[4:].strip()

            # label, reason, direction 추출
            labels = re.findall(r'"label"\s*:\s*"([^"]+)"', full_text)
            reasons = re.findall(r'"reason"\s*:\s*"([^"]+)"', full_text)
            directions = re.findall(r'"safe_movement_direction"\s*:\s*"([^"]+)"', full_text)

            # 결과 zip
            results = list(zip(labels, reasons, directions))
            return results, full_text
        except requests.RequestException as e:
            rospy.logerr(f"Ollama API 요청 실패: {e}")
            return [], ""

    def spin(self):
        while not rospy.is_shutdown():
            if self.latest_img is not None:
                try:
                    # Ollama VQA 실행
                    results, raw_text = self.run_ollama_vqa_and_extract_info(self.latest_img)
                    if raw_text:
                        # JSON 결과를 /ollama_msg 토픽으로 발행
                        self.pub.publish(String(raw_text))
                        print("Ollama VQA 결과 발행:")
                        for i, (label, reason, direction) in enumerate(results):
                            print(f"[{i}] 라벨: {label}, 이유: {reason}, 방향: {direction}")
                    else:
                        rospy.logwarn("Ollama에서 유효한 응답 없음")
                except Exception as e:
                    rospy.logerr(f"처리 중 에러: {e}")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = OllamaVQANode()
        node.spin()
    except rospy.ROSInterruptException:
        pass