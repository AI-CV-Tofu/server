import numpy as np
import json
import base64
import cv2
import boto3
import torch
import time
from collections import defaultdict
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from .db_utils import insert_tofu_production, insert_defect_details, update_tofu_production
from .db_utils import line_chart, pie_chart, bar_chart
from django.http import StreamingHttpResponse
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)


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
        
        # 모든 defect_type 초기화
        all_defect_types = ["bubble", "chip", "cut", "debris", "dent", "line", "spot"]
        defect_data = {defect: 0 for defect in all_defect_types}

        # bar_chart()에서 반환된 데이터로 counts 업데이트
        for defect, count in zip(defect_types, defect_counts):
            if defect in defect_data:
                defect_data[defect] = count

        # Response 데이터 정의
        response_data = {
            'pie_chart': {
                'OK': OK_count,
                'NG': NG_count
            },
            'bar_chart': {
                'defect_type': list(defect_data.keys()),
                'counts': list(defect_data.values())
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



class DashboardStreamData(APIView):
    def get(self, request):
        """
        실시간 대시보드 데이터 스트리밍
        """
        def event_stream():
            while True:
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
                    
                    logger.info(f"전송된 데이터: {response_data}")  # 디버깅 로그
                    yield f"data: {json.dumps(response_data)}\n\n"
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"SSE 에러: {e}")
                    yield f"event: error\ndata: {str(e)}\n\n"
                    time.sleep(1)

        response = StreamingHttpResponse(
            event_stream(), 
            content_type='text/event-stream'
        )
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Headers'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET'
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
    

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
                return Response({"status": "error", "message": "image_array is None. Check the request payload."}, status=400)
            if image_array.size == 0:
                return Response({"status": "error", "message": "image_array is empty. Check the input data."}, status=400)
            if len(image_array.shape) < 3:
                return Response({"status": "error", "message": f"Invalid image dimensions: {image_array.shape}. Expected 3 dimensions."}, status=400)

            # 디버깅 로그 출력
            print("Received image_array:", image_array)
            print("Type of image_array:", type(image_array))
            print("Shape of image_array:", image_array.shape)
            print("Data type of image_array:", image_array.dtype)
            
            # 3. Tofu_Production에 새로운 행 추가
            tofu_id = insert_tofu_production()
            if tofu_id is None:
                raise Exception("Failed to insert into Tofu_Production")

            # 4. SageMaker 모델 실행 (병렬 처리)
            model_results = self.run_models(image_array)

            # 5. 결과 결합 및 NMS 적용
            final_result = self.merge_results(model_results)

            # 6. 모델 결과를 Defect_Details에 저장
            final_boxes = final_result["boxes"]
            final_scores = final_result["scores"]
            final_classes = final_result["classes"]
            defect_types = final_result["defect_types"]

            defects = []
            all_classes_are_cut = True  # 모든 클래스가 2인지 확인
            if final_boxes and len(final_boxes) > 0:
                for i, box in enumerate(final_boxes):
                    defect = {
                        "type": defect_types[i],
                        "box": box,
                        "score": round(final_scores[i], 3),
                        "class": final_classes[i]
                    }
                    defects.append(defect)

                    # 모든 클래스가 2인지 확인
                    if final_classes[i] != 2:
                        all_classes_are_cut = False

                    # Defect_Details에 저장
                    bounding_box = ",".join(map(str, box))
                    insert_defect_details(tofu_id, defect["type"], bounding_box)

                # 결과가 있을 경우 defect_status를 1로 업데이트
                update_tofu_production(tofu_id, detection_status=1, defect_status=1)
            else:
                # 결과가 없을 경우 defect_status는 0으로 유지하고 detection_status만 1로 업데이트
                update_tofu_production(tofu_id, detection_status=1, defect_status=0)

            # 결함 상태 설정
            if len(defects) == 0:
                defect_status = "OK"
            elif all_classes_are_cut:
                defect_status = "OK"
            else:
                defect_status = "NG"

            # 응답 데이터 구성
            return Response({
                "status": "success",
                "message": "Processing completed",
                "results": {
                    "defect_status": defect_status,  # defect_status 먼저
                    "defects": defects  # defects 뒤로 이동
                }
            }, status=200)

        except Exception as e:
            return Response({"status": "error", "message": str(e)}, status=500)

    def run_models(self, image_array):
        """
        두 SageMaker 모델을 병렬로 호출하여 결과 반환
        """
        resized_image = cv2.resize(image_array, (640, 640))  # 크기 조정
        resized_jpeg = cv2.imencode('.jpg', resized_image)[1]
        payload = base64.b64encode(resized_jpeg).decode('utf-8')

        endpoints = ["square-tofu-v5", "square-tofu-v4"]
        runtime = boto3.client('runtime.sagemaker')

        def invoke_endpoint(endpoint):
            response = runtime.invoke_endpoint(
                EndpointName=endpoint,
                ContentType='text/csv',
                Body=payload
            )
            response_body = response['Body'].read()
            return json.loads(response_body.decode('ascii'))

        # 병렬로 모델 호출
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(invoke_endpoint, endpoints))

        return results

    def merge_results(self, model_results):
        """
        두 모델의 결과를 결합하고 NMS 적용
        """
        custom_labels = ["bubble", "chip", "cut", "debris", "dent", "line", "spot"]
        combined_boxes = []
        combined_scores = []
        combined_classes = []
        combined_defect_types = []

        # 결과 결합
        for result in model_results:
            if 'boxes' in result:
                for box in result['boxes']:
                    combined_boxes.append(box[:4])
                    combined_scores.append(box[4])
                    combined_classes.append(int(box[-1]))
                    combined_defect_types.append(custom_labels[int(box[-1])])

        # NMS 적용 (Non-Maximum Suppression)
        final_boxes, final_scores, final_classes = self.apply_nms(combined_boxes, combined_scores, combined_classes)
        final_defect_types = [custom_labels[cls] for cls in final_classes]

        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
            "defect_types": final_defect_types
        }

    def apply_nms(self, boxes, scores, classes, threshold=0.5):
        """
        Non-Maximum Suppression(NMS)을 적용하여 중복 박스를 제거
        """
        indices = torch.argsort(torch.tensor(scores), descending=True)
        selected_indices = []

        for i in indices:
            keep = True
            for j in selected_indices:
                if self.is_overlap(boxes[i], boxes[j], threshold):
                    keep = False
                    break
            if keep:
                selected_indices.append(i)

        final_boxes = [boxes[i] for i in selected_indices]
        final_scores = [scores[i] for i in selected_indices]
        final_classes = [classes[i] for i in selected_indices]

        return final_boxes, final_scores, final_classes

    def is_overlap(self, box1, box2, threshold=0.5):
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