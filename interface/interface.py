import streamlit as st
import io
import requests
import argparse
from cv2 import imread, cvtColor, COLOR_BGR2RGB, imencode, imdecode, IMREAD_COLOR
import cv2
import numpy as np
import base64
from io import BytesIO

def interpret(response):
    if response.status_code == 200:
        # API 응답 결과 해석
        result = response.json()
        category = result["category"]
        video_data = response.content
        
        # 예측 결과 출력
        st.success("완료되었습니다!")
        st.subheader("예측 결과")
        st.write("분류:", category)

        if category == "숫자인식":
            st.write("숫자:", result['prediction'])
            st.write("신뢰도:", result['confidence'])

        elif category == "알파벳인식":
            st.write("글자:", result['prediction'])
            st.write("신뢰도:", result['confidence'])
        
        elif category == "자유패턴":
            st.write("신뢴도:", result['confidence'])
        
        if "r" in result.keys():
            st.write("반색 여부:", result['r'])

        if "video" in result.keys():
            video_bytes = base64.b64decode(result['video'])
            st.subheader("전처리 과정 시각화")
            st.video(BytesIO(video_bytes))

    else:
        st.error("예측 실패했습니다.")

def predict(uploaded_image, predict_r=True, visualize_preprocessing=True):
    # 이미지를 업로드하고 API에 전송
    img = uploaded_image.read()
    img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # 이미지를 바이트 스트림으로 변환
    img_bytes = io.BytesIO()
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes.write(buffer.tobytes())

    # API 요청을 위한 옵션 설정
    options = {
        "predict_r": 1 if predict_r else 0,
        "visualize_preprocessing": 1 if visualize_preprocessing else 0
    }

    # API 응답 받기
    response = requests.post("http://" + args.api_address + ":" + args.port + "/predict/", files={"file": img_bytes.getvalue()}, data=options)
    return response

def main():
    st.set_page_config(
        page_title="장난감 블록 이미지 분류기",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("장난감 블록 이미지 분류기")
    st.markdown("### 이미지를 업로드하면 장난감 블록 이미지를 분류해줍니다.")

    # 설정 옵션
    with st.sidebar:
        st.header("설정")
        predict_r = st.checkbox("반색 여부 예측", value=False, key="predict_r")
        visualize_preprocessing = st.checkbox("이미지 전처리 시각화", value=False, key="visualize_preprocessing")

    uploaded_image = st.file_uploader("파일을 선택해주세요...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="업로드된 이미지", use_column_width=True)

        if st.button("예측하기"):
            with st.spinner("예측 중..."):
                response = predict(uploaded_image, predict_r, visualize_preprocessing)
                interpret(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--api_address", type=str, default="localhost", help="API 주소")
    parser.add_argument("-p", "--port", type=str, default="8000", help="API 포트")
    args = parser.parse_args()
    main()