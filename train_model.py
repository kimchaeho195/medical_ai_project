"""
train_model.py
----------------
모달리티(X-ray / CT / MRI)별로 질환 분류 AI를 학습시키는 스크립트.

[사용법]
1. 아래 폴더 구조로 이미지를 정리한다.

data/
  xray/
    normal/           <- 정상 X-ray 이미지들
    pneumonia/         <- 폐렴 X-ray 이미지들
    fracture/           <- 골절 X-ray 이미지들
  ct/
    normal/
    hemorrhage/         <- 급성 뇌출혈 CT 이미지들
    cancer/             <- 암(폐암 등) CT 이미지들
  mri/
    normal/
    disc_herniation/    <- 허리디스크 MRI 이미지들

2. 터미널에서 다음처럼 실행 (모달리티 이름만 바꿔가며 3번 실행):

   python train_model.py --modality xray
   python train_model.py --modality ct
   python train_model.py --modality mri

3. 학습이 끝나면 models/xray_model.keras, models/ct_model.keras, models/mri_model.keras
   그리고 각 모델의 클래스 순서를 담은 models/xray_classes.json 등이 생성된다.
   이 파일들을 app.py(추론 서버)가 읽어서 사용한다.

[주의 - 교육용 프로젝트 필독]
- 이 모델은 학교 탐구활동(사람 판독 vs AI 판독 비교) 목적으로만 사용해야 하며,
  실제 환자 진단이나 의료 행위에 절대 사용해서는 안 된다.
- 인터넷에서 수집한 이미지는 저작권/초상권/개인정보(환자 식별정보 포함 여부)를
  반드시 확인하고, 발표/보고서에는 출처를 명시하는 것이 좋다.
"""

import argparse
import json
import os
from collections import Counter

import tensorflow as tf
from PIL import ImageFile
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 일부 이미지 파일이 살짝 잘려있거나 손상되어 있어도 학습이 중단되지 않도록 설정
# (인터넷에서 대량으로 모은 이미지 중 일부는 다운로드 과정에서 손상될 수 있음)
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 15          # 1단계: 마지막 분류층만 학습
FINE_TUNE_EPOCHS = 15     # 2단계: 일부 층을 풀어서 추가로 미세조정(파인튜닝)
FINE_TUNE_UNFREEZE_LAYERS = 30  # MobileNetV2 마지막 30개 층을 풀어서 같이 학습
FINE_TUNE_LR = 1e-5        # 파인튜닝은 아주 작은 학습률로 조심스럽게


def build_model(num_classes: int):
    """MobileNetV2 기반 전이학습 모델을 만든다.
    1단계에서는 특징 추출부를 얼려두고 마지막 분류층만 학습시키고,
    2단계(파인튜닝)에서 일부 층을 풀어서 정확도를 더 끌어올린다.

    주의: 이미지 전처리(preprocess_input)는 모델 내부가 아니라
    데이터 생성기(ImageDataGenerator)에서 처리한다. 모델 안에 원시 TF 연산을
    직접 넣으면 나중에 모델을 불러올 때 "Unknown layer" 오류가 발생하기 때문이다.
    """
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # 1단계: 사전학습된 특징 추출부는 고정

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train(modality: str, data_dir: str, output_dir: str):
    modality_dir = os.path.join(data_dir, modality)
    if not os.path.isdir(modality_dir):
        raise FileNotFoundError(
            f"'{modality_dir}' 폴더가 없어. data/{modality}/정상/질환1/질환2 구조로 이미지를 넣어줘."
        )

    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2,
        rotation_range=8,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(
        modality_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        modality_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    class_indices = train_gen.class_indices  # 예: {'normal': 0, 'pneumonia': 1, ...}
    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"[{modality}] 클래스: {class_indices}")

    # 클래스별 이미지 개수가 다르면(예: 정상 500장, 질환 100장) AI가 그냥
    # 많은 쪽만 찍어도 정확도가 높아 보이는 착시가 생긴다. 이를 막기 위해
    # 이미지 수가 적은 클래스에 더 큰 가중치를 자동으로 부여한다.
    class_counts = Counter(train_gen.classes)
    total_count = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weight = {
        cls_idx: total_count / (num_classes * count)
        for cls_idx, count in class_counts.items()
    }
    print(f"[{modality}] 클래스별 이미지 수: "
          f"{ {idx_to_class[i]: c for i, c in class_counts.items()} }")
    print(f"[{modality}] 자동 계산된 클래스 가중치: "
          f"{ {idx_to_class[i]: round(w, 2) for i, w in class_weight.items()} }")

    model, base_model = build_model(num_classes=num_classes)

    # ===== 1단계: 분류층만 학습 =====
    print(f"\n===== [{modality}] 1단계 학습 시작 (특징 추출부 고정) =====")
    head_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=HEAD_EPOCHS,
        callbacks=head_callbacks,
        class_weight=class_weight,
    )

    # ===== 2단계: 파인튜닝 (일부 층을 풀어서 추가 학습) =====
    print(f"\n===== [{modality}] 2단계 파인튜닝 시작 (마지막 {FINE_TUNE_UNFREEZE_LAYERS}개 층 학습) =====")
    base_model.trainable = True
    # 마지막 N개 층만 학습 가능하게 풀고, 나머지는 그대로 고정해둔다
    for layer in base_model.layers[:-FINE_TUNE_UNFREEZE_LAYERS]:
        layer.trainable = False

    # 파인튜닝 단계는 학습률을 훨씬 낮춰서 기존에 배운 특징이 망가지지 않게 한다
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=fine_tune_callbacks,
        class_weight=class_weight,
    )

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{modality}_model.keras")
    classes_path = os.path.join(output_dir, f"{modality}_classes.json")

    model.save(model_path)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {model_path}")
    print(f"클래스 정보 저장: {classes_path}")

    # 검증 데이터 최종 정확도 출력 (STEP 25 결과 정리에 바로 활용 가능)
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n[{modality}] 최종 검증 정확도: {val_acc:.4f}, 검증 손실: {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        required=True,
        choices=["xray", "ct", "mri"],
        help="학습할 영상 종류",
    )
    parser.add_argument("--data_dir", default="data", help="데이터 최상위 폴더")
    parser.add_argument("--output_dir", default="models", help="모델 저장 폴더")
    args = parser.parse_args()

    train(args.modality, args.data_dir, args.output_dir)
