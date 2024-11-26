import requests
import cv2
import os

# API URL
url = "http://44.214.252.225:8000/process-image/"

# 1. 이미지 파일 읽기
file_path = os.path.join(os.getcwd(), '4748.jpg')  # 테스트용 이미지 경로
image_array = cv2.imread(file_path)  # OpenCV로 이미지 읽기

# 2. 유효성 검사
if image_array is None:
    raise ValueError(f"Unable to load image from {file_path}.")

# 3. API 요청
headers = {"Content-Type": "application/json"}  # JSON 형태로 전송
payload = {"image": image_array.tolist()}  # 넘파이 배열을 리스트로 변환
response = requests.post(url, json=payload, headers=headers)

# 4. 응답 출력
print("Status Code:", response.status_code)
print("Response:", response.json())
