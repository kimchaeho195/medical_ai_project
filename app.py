"""
app.py
-------
학습된 모델(xray_model.h5, ct_model.h5, mri_model.h5)을 불러와서
웹 플랫폼에서 이미지를 보내면 AI 판독 결과(JSON)를 돌려주는 서버.

[사용법]
1. models/ 폴더에 train_model.py로 만든 .h5, _classes.json 파일들이 있어야 함
2. 터미널에서 실행:
     python app.py
   -> http://localhost:5000 에서 서버가 뜬다.

3. 플랫폼(React/Next.js 등) 프론트엔드에서는 이미지를 아래처럼 보내면 된다:

   const formData = new FormData();
   formData.append("image", file);          // 업로드한 이미지 파일
   formData.append("modality", "xray");      // "xray" / "ct" / "mri" 중 하나

   const res = await fetch("http://localhost:5000/predict", {
     method: "POST",
     body: formData,
   });
   const result = await res.json();
   // result 예시:
   // {
   //   "modality": "xray",
   //   "prediction": "pneumonia",
   //   "confidence": 0.91,
   //   "all_probabilities": { "normal": 0.05, "pneumonia": 0.91, "fracture": 0.04 },
   //   "disclaimer": "이 결과는 참고용이며 실제 진단이 아닙니다."
   // }

4. 이 result를 그대로 STEP 20(결과 비교 표)이나 기능 1(AI-사람 비교),
   기능 3(신뢰도 표시)에 사용하면 된다.
"""

import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # 플랫폼(다른 포트/도메인)에서 API를 호출할 수 있게 허용

MODEL_DIR = "models"
IMG_SIZE = (224, 224)

# 서버 시작 시 세 모델을 미리 로드해둔다 (요청마다 로드하면 느림)
_loaded_models = {}
_loaded_classes = {}


def load_all_models():
    for modality in ["xray", "ct", "mri"]:
        model_path = os.path.join(MODEL_DIR, f"{modality}_model.keras")
        classes_path = os.path.join(MODEL_DIR, f"{modality}_classes.json")
        if os.path.exists(model_path) and os.path.exists(classes_path):
            _loaded_models[modality] = load_model(model_path)
            with open(classes_path, "r", encoding="utf-8") as f:
                _loaded_classes[modality] = json.load(f)
            print(f"[로드 완료] {modality} 모델")
        else:
            print(f"[경고] {modality} 모델이 없음. 먼저 train_model.py로 학습해줘.")


def preprocess_image(file_storage) -> np.ndarray:
    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    # 학습 때와 똑같은 전처리를 적용해야 정확한 결과가 나온다
    # (train_model.py에서 ImageDataGenerator가 자동으로 해주던 것을 여기서는 직접 적용)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    array = np.expand_dims(array, axis=0)  # (1, 224, 224, 3)
    return array


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image 파일이 없습니다."}), 400

    modality = request.form.get("modality")
    if modality not in ["xray", "ct", "mri"]:
        return jsonify({"error": "modality는 xray, ct, mri 중 하나여야 합니다."}), 400

    if modality not in _loaded_models:
        return jsonify({"error": f"{modality} 모델이 아직 학습되지 않았습니다."}), 400

    image_file = request.files["image"]
    img_array = preprocess_image(image_file)

    model = _loaded_models[modality]
    idx_to_class = _loaded_classes[modality]

    probabilities = model.predict(img_array)[0]  # shape: (num_classes,)
    pred_idx = int(np.argmax(probabilities))
    pred_class = idx_to_class[str(pred_idx)]
    confidence = float(probabilities[pred_idx])

    all_probs = {
        idx_to_class[str(i)]: round(float(p), 4) for i, p in enumerate(probabilities)
    }

    return jsonify(
        {
            "modality": modality,
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "disclaimer": "이 결과는 학교 탐구활동용 참고 정보이며 실제 의료 진단이 아닙니다.",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded_models": list(_loaded_models.keys())})


@app.route("/debug", methods=["GET"])
def debug():
    """임시 디버그용: 서버가 실제로 보고 있는 폴더 구조를 확인한다.
    문제 해결 후에는 이 라우트를 지워도 된다."""
    cwd = os.getcwd()
    root_files = os.listdir(".")
    models_files = os.listdir("models") if os.path.isdir("models") else "models 폴더 없음"
    return jsonify({
        "현재_작업_폴더": cwd,
        "최상위_파일목록": root_files,
        "models_폴더_안": models_files,
    })


if __name__ == "__main__":
    load_all_models()
    # Render 같은 배포 서비스는 PORT 환경변수로 포트 번호를 알려준다.
    # 로컬 컴퓨터에서 실행할 땐 환경변수가 없으니 그냥 5000번을 쓴다.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
