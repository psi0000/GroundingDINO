#!/usr/bin/env python3
import rospy
import json
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# GroundingDINO inference util
from groundingdino.util.inference import load_model, predict, annotate

class GroundingDINORosNode:
    def __init__(self):
        rospy.init_node('groundingdino_node')

        # Hardcoded config and weight paths
        cfg_path = "../../groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights_path = "../../weights/groundingdino_swint_ogc.pth"
        self.model = load_model(cfg_path, weights_path)

        # Thresholds
        self.box_thresh = 0.35
        self.text_thresh = 0.25

        # Bridge and state
        self.bridge = CvBridge()
        self.latest_img = None
        self.latest_prompt = None

        # Subscribers
        #TODO: Change topic names
        rospy.Subscriber('/raw_image', Image,
                         self._image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/ollama_msg', String,
                         self._prompt_cb, queue_size=1)

        # Publishers
        self.box_pub = rospy.Publisher('/groundingdino/boxes', String, queue_size=1)
        self.img_pub = rospy.Publisher('/groundingdino/annotated_image', Image, queue_size=1)

        # Wait for first image
        rospy.wait_for_message('/raw_image', Image)
        self.rate = rospy.Rate(10)  # 10 Hz

    def _prompt_cb(self, msg: String):
        """Update current text prompt."""
        self.latest_prompt = msg.data

    def _image_cb(self, msg: Image):
        """Receive raw image."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_img = img
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def spin(self):
        while not rospy.is_shutdown():
            if self.latest_img is not None and self.latest_prompt:
                # 1) GroundingDINO inference
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=self.latest_img,
                    caption=self.latest_prompt,
                    box_threshold=self.box_thresh,
                    text_threshold=self.text_thresh
                )

                # 2) Publish boxes as JSON string
                results = [
                    {'box': b.tolist(), 'score': float(s), 'phrase': p}
                    for b, s, p in zip(boxes, logits, phrases)
                ]
                self.box_pub.publish(String(json.dumps(results)))

                # 3) Annotate image and publish
                annotated = annotate(
                    image_source=self.latest_img.copy(),
                    boxes=boxes, logits=logits, phrases=phrases
                )
                try:
                    out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
                    out_msg.header.stamp = rospy.Time.now()
                    out_msg.header.frame_id = 'camera'
                    self.img_pub.publish(out_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"Publish Error: {e}")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = GroundingDINORosNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
