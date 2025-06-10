import requests
import base64
import json
import re

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_ollama_vqa_and_extract_info(image_path, model_name="qwen2.5vl"):
    prompt = """
    You will be given an image from a drone's onboard camera flying indoors.

    Your task is to identify specific areas that a drone should avoid flying toward. These are not necessarily physical obstacles, but could still be dangerous due to:

    - critical safety functions (e.g., alarms, fire extinguishers),
    - unpredictable behavior (e.g., open doors, hanging objects),
    - semantically ambiguous regions that RGB sensors may misinterpret (e.g., reflections, glass, shadows)

    For each such region:
    - Provide a short label describing the object or area
    - Explain why it is dangerous
    - Suggest a **safe movement direction** based on the drone's current viewpoint, relative to the danger region. Use one of the following expressions: "move left", "move right", "move up", or "move down".

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
        "risk region position": {
            "bbox_2d": [x_min, y_min, x_max, y_max]
        }
        }
    ]
    }
    """

    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_b64],
        "stream": True
    }

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

# 🧪 실행
if __name__ == "__main__":
    image_path = "/home/psi/Desktop/asl/3D_vision/BLIP/test2.png"  # 🔁 여길 이미지 경로로 바꿔도 됩니다
    results, raw_text = run_ollama_vqa_and_extract_info(image_path)

    print("🧠 Extracted danger zones:")
    for i, (label, reason, direction) in enumerate(results):
        print(f"[{i}] 🔖 Label: {label}\n    ⚠️ Reason: {reason}\n    🧭 Direction: {direction} \n\n")
    
    # 디버깅용 전체 출력 확인
    # print("\n📄 Raw Output:\n", raw_text)