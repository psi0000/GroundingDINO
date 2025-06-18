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
import time
class OllamaVQANode:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('ollama_vqa_node')
        # CvBridge 초기화
        self.bridge = CvBridge()
        self.resize_to = (320, 240) 
        self.latest_img = None
        # ROS 설정
        rospy.Subscriber('/rgb_image', Image, self._image_cb, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher('/ollama_msg', String, queue_size=1)
        # Ollama 모델 이름
        self.model_name = "qwen2.5vl"
        # 프롬프트 정의
        self.prompt = """
            An onboard camera image captured during an indoor drone flight is provided.

            Your task is to identify step-by-step **only those regions in the image that present a semantic risk** to the drone’s safe flight.  

            ### Definitions

            - **Explicit risk**: A clearly visible object or area whose risk comes from its state, properties, or context—**not just its physical presence**.
            - *Examples:*
                - A hanging object that looks unstable or might fall
                - An open door, because something or someone could appear from behind it
                - A window that is open to an unknown or dangerous environment

            - **Implicit risk**: A region or situation where risk is suggested by ambiguity, occlusion, or limited visibility—**but only if there are visible cues in the image**.
            - *Examples:*
                - An area partially blocked by another object (possibly hiding a hazard)
                - A door that suggests a hidden fire
                - Broken window
            - `"risk_direction"`: The direction in which the **drone should move to safely avoid the detected risk object or area**.
                - If the risk is located on the right side of the drone, then `"risk_direction"` should be `"left"` (meaning: move left to avoid the risk).
                - If the risk is above the drone (e.g., a hanging object), then `"risk_direction"` should be `"down"` (do not fly below it because falling possible).
                - If the risk is on the left, use `"right"`.

            If possible, select the most intuitive direction to avoid direct collision or danger, based on the drone's likely flight path.
            **Do NOT report risks that are not visually present or supported by cues in the image.**
            - **Must Do NOT include label such as plain walls and ceiling
            - **Do NOT hallucinate or imagine hidden risks.** Only report risks that have visible evidence in the image (including visible cues, ambiguous regions, or clear context).


            For each detected danger zone, report:
            - `"risk_type"`: "explicit" or "implicit"
            - `"label"`:  "[state]+[risk]" or "[semantic risk description]" (e.g., "hanging object", "broken window", "open door", "fire", etc.)
            - `"reason"`: why it is dangerous , explain risk position.
            - `"risk_direction"`: (choose one: up, down, left, right)
            
            Respond **strictly** in the following JSON format (English only):

            ```json
            {
            "danger_zones": [
                {
                "risk_type": "explicit | implicit",
                "label": "...",
                "reason": "...",
                "risk_direction": "up | down | left | right"
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
        # === 1. 이미지 리사이즈 추가 ===
        if self.resize_to is not None:
            resized_image = cv2.resize(image, self.resize_to)
        else:
            resized_image = image

        # === 2. Base64 인코딩에 리사이즈된 이미지 사용 ===
        image_b64 = self.encode_image_to_base64(resized_image)

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
            risk_types = re.findall(r'"risk_type"\s*:\s*"([^"]+)"', full_text)
            labels = re.findall(r'"label"\s*:\s*"([^"]+)"', full_text)
            reasons = re.findall(r'"reason"\s*:\s*"([^"]+)"', full_text)
            directions = re.findall(r'"risk_direction"\s*:\s*"([^"]+)"', full_text)


            # 결과 zip
            results = list(zip(risk_types, labels, reasons, directions))
            return results, full_text
        except requests.RequestException as e:
            rospy.logerr(f"Ollama API 요청 실패: {e}")
            return [], ""

    def spin(self):
        while not rospy.is_shutdown():
            if self.latest_img is not None:
                try:
                    print("Ollama VQA 요청 시작...")
                    start_time = time.time()
                    # Ollama VQA 실행
                    results, raw_text = self.run_ollama_vqa_and_extract_info(self.latest_img)
                    end_time = time.time()  # 종료 시간
                    duration = end_time - start_time  # 실행 시간 계산
                    print(f"Ollama VQA 실행 시간: {duration:.2f}초")
                    if raw_text:
                        # JSON 결과를 /ollama_msg 토픽으로 발행
                        self.pub.publish(String(raw_text))
                        print("Ollama VQA 결과 발행:")
                        for i, (risk_type, label, reason, direction) in enumerate(results):
                            print(f"[{i}] 유형: {risk_type}, 라벨: {label}, 이유: {reason}, 방향: {direction}")
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