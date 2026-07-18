"""
quick_update.py
----------------
이미 학습된 모델(models/*.keras)에 새 이미지 한 장을 빠르게 추가로 학습시키는 스크립트.
train_model.py처럼 전체 데이터를 다시 학습하는 게 아니라서 몇 초~몇 분이면 끝난다.

⚠️ 꼭 읽어줘 (중요한 주의사항)
이 방법은 "정석적인 정확도 개선 방법"이 아니다. 사진 한두 장만 추가로 학습시키면
AI가 그 사진 자체를 외워버릴 뿐이지, 비슷한 다른 사진들까지 더 잘 맞추게 되는 건
아니다 (일종의 "벼락치기"에 가깝다). 그래서 이걸 여러 번 반복해도 실제 성능이
좋아졌다고 보긴 어렵다. 진짜 정확도를 올리려면 결국 데이터를 더 많이 모아서
train_model.py로 처음부터(또는 파인튜닝까지) 다시 학습시키는 게 정석이다.

그래도 특정 사진 하나를 빠르게 다시 가르치고 싶을 때 (예: 발표 시연에 쓸 사진을
AI가 계속 틀리게 맞출 때) 이 스크립트를 쓰면 된다.

[사용법 예시]
python quick_update.py --modality xray --image "C:\\Users\\me\\Desktop\\내사진.jpg" --label pneumonia

--label 에는 학습할 때 썼던 폴더 이름을 그대로 적어야 한다.
(xray: normal, pneumonia, fracture / ct: normal, hemorrhage, cancer / mri: normal, disc_herniation)
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = (224, 224)
QUICK_EPOCHS = 8
QUICK_LR = 1e-5  # 기존에 배운 내용이 무너지지 않도록 아주 작은 학습률 사용


def load_image_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["xray", "ct", "mri"])
    parser.add_argument("--image", required=True, help="추가로 학습시킬 이미지 파일 경로")
    parser.add_argument("--label", required=True, help="이 이미지의 정답 클래스 이름")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, f"{args.modality}_model.keras")
    classes_path = os.path.join(args.model_dir, f"{args.modality}_classes.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"'{model_path}'가 없어. 먼저 train_model.py로 학습부터 해야 해.")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"'{args.image}' 이미지 파일을 찾을 수 없어. 경로를 다시 확인해줘.")

    with open(classes_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)  # 예: {"0": "normal", "1": "pneumonia"}
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}

    if args.label not in class_to_idx:
        raise ValueError(
            f"'{args.label}'은(는) 알 수 없는 클래스야. 가능한 값: {list(class_to_idx.keys())}"
        )

    print(f"모델 불러오는 중: {model_path}")
    model = tf.keras.models.load_model(model_path)

    img_array = load_image_array(args.image)
    x = np.expand_dims(img_array, axis=0)

    num_classes = len(class_to_idx)
    y = tf.keras.utils.to_categorical([class_to_idx[args.label]], num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=QUICK_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"'{args.label}' 라벨로 {QUICK_EPOCHS}번 추가 학습 중...")
    model.fit(x, y, epochs=QUICK_EPOCHS, verbose=1)

    model.save(model_path)
    print(f"\n저장 완료: {model_path}")
    print("⚠️ 참고: 이 사진 한 장에 대한 결과만 바뀐 것일 수 있고, 전체 성능이 좋아졌다는 뜻은 아니야.")
    print("   (AI 서버가 켜져있다면 Ctrl+C로 껐다가 다시 켜야 이 변경된 모델을 인식해)")


if __name__ == "__main__":
    main()
