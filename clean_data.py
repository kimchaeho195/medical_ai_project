"""
clean_data.py
----------------
data 폴더 안의 모든 이미지를 하나씩 열어보고,
손상되었거나 열리지 않는 파일은 자동으로 찾아서 삭제해주는 스크립트.

[사용법]
python clean_data.py       (또는 py -3.13 clean_data.py)

학습(train_model.py) 도중 "image file is truncated" 같은 에러가 나면
이 스크립트를 먼저 실행해서 깨진 파일들을 정리한 다음 다시 학습을 시도하면 된다.
"""

import os

from PIL import Image, ImageFile

# 살짝 손상된 이미지는 강제로 읽기 시도 (완전히 깨진 것만 걸러내기 위함)
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = "data"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def is_image_ok(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()  # 실제로 픽셀까지 읽어봐야 완전히 확인됨
        return True
    except Exception:
        return False


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"'{DATA_DIR}' 폴더를 찾을 수 없습니다. medical_ai_project 폴더 안에서 실행해주세요.")
        return

    total_checked = 0
    total_removed = 0

    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            filepath = os.path.join(root, filename)

            # 이미지 확장자가 아닌 파일(예: .txt, desktop.ini 등)도 삭제 대상
            if not filename.lower().endswith(VALID_EXTENSIONS):
                print(f"[삭제] 이미지 파일이 아님: {filepath}")
                os.remove(filepath)
                total_removed += 1
                continue

            total_checked += 1
            if not is_image_ok(filepath):
                print(f"[삭제] 손상된 이미지: {filepath}")
                os.remove(filepath)
                total_removed += 1

    print(f"\n검사 완료: 총 {total_checked}개 이미지 확인, {total_removed}개 삭제됨.")
    print("이제 train_model.py를 다시 실행해도 좋습니다.")


if __name__ == "__main__":
    main()
