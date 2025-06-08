#!/usr/bin/env python3
import rospy
import requests
import base64
import json
import re
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class OllamaRosNode:
    def __init__(self):
        rospy.init_node('ollama_ros_node')
        self.bridge = CvBridge()
        self.latest_img = None
        self.msg_pub = rospy.Publisher('/ollama_msg', String, queue_size=1)
        rospy.Subscriber('/raw_image', Image, self._image_cb, queue_size=1, buff_size=2**24)
        rospy.wait_for_message('/raw_image', Image)
        self.rate = rospy.Rate(1)

    def _image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_img = cv_img
        except CvBridgeError as e:
            rospy.logerr(f"[ollama] CvBridge error: {e}")

    def encode_image_to_base64(self, cv_img):
        _, buffer = cv2.imencode('.png', cv_img)
        return base64.b64encode(buffer).decode('utf-8')

    def run_ollama_vqa_and_extract_info(self, cv_img, model_name="qwen2.5vl"):
        prompt = """
        You will be given an image from a drone's onboard camera flying indoors.

        Your task is to identify specific areas that a drone should avoid flying toward. These are not necessarily physical obstacles, but could still be dangerous due to:

        - critical safety functions (e.g., alarms, fire extinguishers),
        - unpredictable behavior (e.g., open doors, hanging objects),
        - semantically ambiguous regions that RGB sensors may misinterpret (e.g., reflections, glass, shadows)

        For each such region:
        - Provide a short label describing the object or area
        - Explain why it is dangerous
        - Suggest a **safe movement direction** based on the danger_zones, relative to the danger region. Use one of the following expressions: "move left", "move right", "move up", or "move down".

        Also include:
        - The internal_image_resolution used for processing. If the image was not resized, return "original"

        Respond only in the following strict JSON format:

        {
        "internal_image_resolution": "original",
        "danger_zones": [
            {
            "label": "concise description of the region",
            "reason": "why it is dangerous",
            "safe_movement_direction": "move left | move right | move up | move down",
            }
        ]
        }
        """
        img_b64 = self.encode_image_to_base64(cv_img)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": True
        }
        resp = requests.post('http://localhost:11434/api/generate', json=payload, stream=True, timeout=60)
        resp.raise_for_status()
        full_text = ''
        for line in resp.iter_lines():
            if line:
                try:
                    part = json.loads(line.decode('utf-8'))
                    full_text += part.get('response', '')
                except Exception:
                    continue
        full_text = full_text.strip('`\n ')
        if full_text.lower().startswith('json'):
            full_text = full_text[4:].strip()
        return full_text

    def spin(self):
        while not rospy.is_shutdown():
            if self.latest_img is not None:
                try:
                    result_json = self.run_ollama_vqa_and_extract_info(self.latest_img)
                    data = json.loads(result_json)
                    zones = data.get('danger_zones', [])
                    output = [
                        {
                            'label': z.get('label', ''),
                            'direction': z.get('safe_movement_direction', '')
                        }
                        for z in zones
                    ]
                    publish_str = json.dumps(output)
                except Exception as e:
                    rospy.logerr(f"[ollama] Error parsing/resulting data: {e}")
                    publish_str = '[]'
                self.msg_pub.publish(String(data=publish_str))
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = OllamaRosNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
