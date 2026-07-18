# 의료영상 AI 판독 모델 - 사용 가이드

## 1. 전체 구조

```
data/                    <- 학습용 이미지 (직접 준비)
  xray/normal, xray/pneumonia, xray/fracture
  ct/normal, ct/hemorrhage, ct/cancer
  mri/normal, mri/disc_herniation

train_model.py            <- 모델 학습 스크립트
app.py                     <- 학습된 모델을 API로 서빙 (플랫폼과 연결)
requirements.txt
```

## 2. 설치

```bash
pip install -r requirements.txt
```

## 3. 데이터 준비 (방사선 학생 담당)

각 폴더에 이미지를 넣는다. 클래스(질환)당 최소 30~50장 이상 있으면 결과가 안정적이다.
질환당 이미지가 너무 적으면(10장 이하) 정확도가 낮게 나올 수 있으니,
이 경우 보고서에 "데이터 수 제한으로 인한 정확도 한계"로 솔직하게 기록하는 것도
좋은 탐구 포인트가 된다 (STEP 21 오답 분석과 연결됨).

## 4. 모델 학습 (AI 학생 담당)

```bash
python train_model.py --modality xray
python train_model.py --modality ct
python train_model.py --modality mri
```

각 명령이 끝나면 `models/` 폴더에 `.h5` 모델 파일과 `_classes.json`이 생긴다.
학습 로그에 마지막으로 뜨는 `검증 정확도`를 캡처해두면 STEP 25(결과 정리),
발표 PPT의 "AI 성능" 슬라이드에 바로 쓸 수 있다.

## 5. API 서버 실행 (AI 학생 담당)

```bash
python app.py
```

`http://localhost:5000/predict` 로 이미지를 보내면 판독 결과를 JSON으로 받는다.
`app.py` 안에 프론트엔드 연동 예시 코드(JavaScript)가 주석으로 들어 있다.

## 6. 이미 만든 플랫폼과 연결하는 법

이미 만든 플랫폼의 프론트엔드 코드(React/Next.js든 다른 것이든)를 보내주면,
이미지 업로드 버튼을 누르면 위 API를 호출해서 결과를 화면에 뿌려주는 부분을
그 코드에 맞춰 정확히 작성해줄게. 지금은 어떤 코드에도 붙을 수 있는
범용 API 형태(`/predict`, FormData 방식)로 만들어뒀다.

## 7. 주의사항 (발표 때 꼭 언급하면 좋은 포인트)

- 이 AI는 교육/탐구 목적이며 실제 진단 도구가 아니다.
- 신뢰도(confidence)는 참고용 수치일 뿐, 확정 진단이 아니다.
- 인터넷에서 수집한 이미지의 출처와 저작권은 참고문헌에 명시한다.
- 데이터가 적을수록 AI가 특정 이미지 특징(워터마크, 촬영 각도 등)에 과적합될 수 있다는
  점은 오답 분석(STEP 21)에서 다루면 좋은 심화 포인트가 된다.
