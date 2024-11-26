import numpy as np
import json
import base64
import cv2
import boto3
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from .db_utils import insert_tofu_production, insert_defect_details, update_tofu_production
from .db_utils import line_chart, pie_chart, bar_chart



@api_view(['GET'])
def dashboard_data(request):
    """
    Dashboard 데이터를 반환하는 API 뷰
    """
    try:
        # Line chart 데이터
        cumulative_OK, cumulative_NG, timestamps = line_chart()
        
        # Pie chart 데이터
        OK_count, NG_count = pie_chart()
        
        # Bar chart 데이터
        defect_types, defect_counts = bar_chart()
        
        # Response 데이터 정의
        response_data = {
            'pie_chart': {
                'OK': OK_count,
                'NG': NG_count
            },
            'bar_chart': {
                'defect_type': defect_types,
                'counts': defect_counts
            },
            'line_chart': {
                'timestamp': timestamps,
                'OK': cumulative_OK,
                'NG': cumulative_NG
            }
        }

        return Response({"status": "success", "data": response_data}, status=200)
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


class ProcessImageAPIView(APIView):
    def post(self, request):
        try:
            # 1. 프론트엔드에서 numpy 배열 받기
            image_array = np.array(request.data.get('image'))
            
            # 2. 데이터 타입 변환
            if image_array.dtype != np.uint8:
                print(f"Converting image_array from {image_array.dtype} to uint8")
                image_array = image_array.astype(np.uint8)  # uint8로 변환
                
            # +. 유효성 검사
            if image_array is None:
                return Response({"error": "image_array is None. Check the request payload."}, status=400)
            if image_array.size == 0:
                return Response({"error": "image_array is empty. Check the input data."}, status=400)
            if len(image_array.shape) < 3:
                return Response({"error": f"Invalid image dimensions: {image_array.shape}. Expected 3 dimensions."}, status=400)

            # 디버깅 로그 출력
            print("Received image_array:", image_array)
            print("Type of image_array:", type(image_array))
            print("Shape of image_array:", image_array.shape)
            print("Data type of image_array:", image_array.dtype)
            
            
            # 3. Tofu_Production에 새로운 행 추가
            tofu_id = insert_tofu_production()
            if tofu_id is None:
                raise Exception("Failed to insert into Tofu_Production")

            # 4. SageMaker 모델 실행
            result = self.run_model(image_array)

            # 5. 모델 결과를 Defect_Details에 저장
            defect_types = []  # 반환할 defect_type 리스트
            if 'boxes' in result and len(result['boxes']) > 0:
                for box in result['boxes']:
                    defect_type = box[-1]  # 라벨 이름
                    bounding_box = ",".join(map(str, box[:4]))
                    insert_defect_details(tofu_id, defect_type, bounding_box)

                    # defect_type을 리스트에 추가
                    defect_types.append(defect_type)

                # 결과가 있을 경우 defect_status를 1로 업데이트
                update_tofu_production(tofu_id, detection_status=1, defect_status=1)
            else:
                # 결과가 없을 경우 defect_status는 0으로 유지하고 detection_status만 1로 업데이트
                update_tofu_production(tofu_id, detection_status=1, defect_status=0)

            # 응답 데이터 구성
            return Response({
                "message": "Processing completed",
                "defect_types": defect_types,  # 최종 defect_type 리스트 반환
                "result": result
            }, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

    def run_model(self, image_array):
        """
        SageMaker 모델 실행 및 결과 반환
        """
        print("Running model with image_array shape:", image_array.shape)
        
        # SageMaker에 보낼 이미지를 준비
        resized_image = cv2.resize(image_array, (640, 640))  # 크기 조정
        resized_jpeg = cv2.imencode('.jpg', resized_image)[1]
        payload = base64.b64encode(resized_jpeg).decode('utf-8')
    
        # SageMaker 호출
        runtime = boto3.client('runtime.sagemaker')
        response = runtime.invoke_endpoint(
            EndpointName="square-tofu-v5",
            ContentType='text/csv',
            Body=payload
        )
        response_body = response['Body'].read()
        result = json.loads(response_body.decode('ascii'))
    
        # 라벨 변환
        custom_labels = ["bubble", "chip", "cut", "debris", "dent", "line", "spot"]
        if 'boxes' in result:
            for box in result['boxes']:
                box[-1] = custom_labels[int(box[-1])]
    
        # NMS를 적용하여 final_scores 계산
        final_boxes = []
        final_scores = []
        final_classes = []
    
        if 'boxes' in result:
            class_boxes = defaultdict(list)
            class_scores = defaultdict(list)
            class_classes = defaultdict(list)
    
            # 클래스별로 박스, 스코어, 클래스 분류
            for box in result['boxes']:
                bounding_box = box[:4]
                score = box[4]
                cls = custom_labels.index(box[-1])  # 라벨 이름을 인덱스로 변환
    
                class_boxes[cls].append(bounding_box)
                class_scores[cls].append(score)
                class_classes[cls].append(cls)
    
            # 겹치는 박스를 제거하기 위한 NMS 적용
            for cls in class_boxes:
                boxes_tensor = torch.tensor(class_boxes[cls])
                scores_tensor = torch.tensor(class_scores[cls])
    
                # 점수 순으로 정렬
                indices = torch.argsort(scores_tensor, descending=True)
    
                # 90% 이상 겹치는 박스 제거
                filtered_boxes = []
                filtered_scores = []
                filtered_classes = []
    
                for idx in indices:
                    box = class_boxes[cls][idx]
                    score = class_scores[cls][idx]
                    label = class_classes[cls][idx]
    
                    # 기존 박스와 비교하여 90% 이상 겹치는 박스 제거
                    keep = True
                    for fbox, fscore, fcls in zip(filtered_boxes, filtered_scores, filtered_classes):
                        if self.is_overlap(box, fbox, threshold=0.9):  # 90% 이상 겹치면 제거
                            if score < fscore:  # confidence가 작은 박스를 제거
                                keep = False
                                break
                    if keep:
                        filtered_boxes.append(box)
                        filtered_scores.append(score)
                        filtered_classes.append(label)
    
                # 최종 결과에 추가
                final_boxes.extend(filtered_boxes)
                final_scores.extend(filtered_scores)
                final_classes.extend(filtered_classes)
    
        print("Final Scores:", final_scores)
        return {"boxes": final_boxes, "scores": final_scores, "classes": final_classes}

    def is_overlap(self, box1, box2, threshold=0.9):
        """
        두 박스가 주어진 threshold 이상 겹치는지 확인
        """
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    
        # 겹치는 영역의 좌표
        ix1 = max(x1, x1_2)
        iy1 = max(y1, y1_2)
        ix2 = min(x2, x2_2)
        iy2 = min(y2, y2_2)
    
        # 겹치는 영역의 넓이 계산
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
        # 각 박스의 넓이 계산
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
        # 겹치는 비율 계산
        overlap_ratio = inter_area / min(box1_area, box2_area)
        return overlap_ratio > threshold