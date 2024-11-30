import mysql.connector
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# MySQL 연결을 설정하는 함수
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="tofu-db.cj24wem202yj.us-east-1.rds.amazonaws.com",  # MySQL 서버 주소
            user="admin",                                            # 사용자 이름
            password="smwu-team-tofu",                               # 비밀번호
            database="tofu_db"                                       # 사용할 데이터베이스 이름
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None



def pie_chart():
    # 데이터베이스 연결
    connection = create_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return None, None  # 연결 실패 시 None 반환

    with connection.cursor() as cursor:
        # OK와 NG의 총 개수를 계산하는 쿼리
        query = """
        SELECT defect_status, COUNT(*) as status_count
        FROM Tofu_Production
        GROUP BY defect_status;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        # OK와 NG 개수 초기화
        OK_count = 0
        NG_count = 0

        # 데이터 처리
        for row in rows:
            defect_status, status_count = row
            if defect_status == 0:  # OK 상태
                OK_count = status_count
            elif defect_status == 1:  # NG 상태
                NG_count = status_count

    # 연결 종료
    connection.close()

    # OK와 NG 개수 반환
    return OK_count, NG_count



def bar_chart():
    # 데이터베이스 연결
    connection = create_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return None, None  # 연결 실패 시 None 반환

    with connection.cursor() as cursor:
        # 각 이미지에서 결함 유형별로 1번만 카운트하는 쿼리
        query = """
        SELECT defect_type, COUNT(DISTINCT tofu_id) as defect_count
        FROM Defect_Details
        GROUP BY defect_type;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        # 결과를 저장할 리스트 초기화
        defect_types = []
        defect_counts = []

        # 데이터 처리
        for row in rows:
            defect_type, defect_count = row
            defect_types.append(defect_type)
            defect_counts.append(defect_count)

    # 연결 종료
    connection.close()

    # 결과 반환
    return defect_types, defect_counts



def line_chart():
    # 데이터베이스 연결
    connection = create_connection()
    if connection is None:
        print("Failed to connect to the database.")
        return None, None, None  # 연결 실패 시 None 반환

    with connection.cursor() as cursor:
        # 데이터베이스에서 시간별 결함 상태 가져오기
        query = """
        SELECT video_frame_timestamp, defect_status
        FROM Tofu_Production
        ORDER BY video_frame_timestamp;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        # 데이터 초기화
        timestamps = []
        cumulative_OK = []
        cumulative_NG = []
        current_OK = 0
        current_NG = 0

        # 데이터 처리
        for row in rows:
            video_frame_timestamp, defect_status = row

            # 타임스탬프 포맷팅
            formatted_timestamp = video_frame_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            timestamps.append(formatted_timestamp)

            # OK/NG 누적합 계산
            if defect_status == 0:
                current_OK += 1
            elif defect_status == 1:
                current_NG += 1

            cumulative_OK.append(current_OK)
            cumulative_NG.append(current_NG)

    # 연결 종료
    connection.close()

    # 누적합과 타임스탬프 반환
    return cumulative_OK, cumulative_NG, timestamps



# Tofu_Production에 데이터 추가
def insert_tofu_production():
    connection = create_connection()
    if connection is None:
        return None
    try:
        with connection.cursor() as cursor:
            query = """
            INSERT INTO Tofu_Production (factory_id, detection_status, defect_status, video_frame_timestamp)
            VALUES (%s, %s, %s, NOW());
            """
            cursor.execute(query, [101, 0, 0])
            connection.commit()
            return cursor.lastrowid  # 새로 추가된 tofu_id 반환
    finally:
        connection.close()

# Defect_Details에 데이터 추가
def insert_defect_details(tofu_id, defect_type, bounding_box):
    connection = create_connection()
    if connection is None:
        return
    try:
        with connection.cursor() as cursor:
            query = """
            INSERT INTO Defect_Details (tofu_id, factory_id, defect_type, bounding_box)
            VALUES (%s, %s, %s, %s);
            """
            cursor.execute(query, [tofu_id, 101, defect_type, bounding_box])
            connection.commit()
    finally:
        connection.close()

# Tofu_Production 업데이트
def update_tofu_production(tofu_id, detection_status, defect_status):
    connection = create_connection()
    if connection is None:
        return
    try:
        with connection.cursor() as cursor:
            query = """
            UPDATE Tofu_Production
            SET detection_status = %s, defect_status = %s
            WHERE tofu_id = %s;
            """
            cursor.execute(query, [detection_status, defect_status, tofu_id])
            connection.commit()
    finally:
        connection.close()