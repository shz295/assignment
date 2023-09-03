from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import os
from cv2 import imdecode, imwrite
import numpy as np
import base64

from preprocessing import preprocess, visualize_transform
from inference import infer, get_category, get_class

app = FastAPI()

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    predict_r: int = Form(0, description="반색 여부 예측"),
    visualize_preprocessing: int = Form(False, description="전처리 과정 시각화")
):

    #try:
    # 업로드된 이미지를 읽어옴
    image_bytes = await file.read()
    image = imdecode(np.frombuffer(image_bytes, np.uint8), 1)


    # 이미지를 전처리하고 추론
    pred, r, confidence = infer(image, return_r=True)

    # 반환할 응답 생성
    response = {
        'category': get_category(pred),
        'confidence': float(confidence)
    }
    
    # 숫자, 알파벳인 경우 예측 결과 반환
    if response['category'] != '자유패턴':
        response['prediction'] = get_class(pred)
    
    # 반색 여부 예측 결과 반환
    if predict_r:
        response['r'] = str(r)

    # 전처리 과정 시각화
    if visualize_preprocessing:
        video_path = "pp.mp4"

        # Create the video and wait for it to finish
        await visualize_transform(image, video_path)

        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found at: {video_path}")

        # Read the video and return it
        with open(video_path, "rb") as video_file:
            encoded_video_bytes = base64.b64encode(video_file.read()).decode("utf-8")

        response['video'] = encoded_video_bytes


    # 결과 반환
    return JSONResponse(content=response, status_code=200)

    #except Exception as e:
    #    return JSONResponse(content={'error': str(e)}, status_code=500)
    
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=str, default="8000", help="API 포트")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host 주소")
    args = parser.parse_args()

    # 서버 실행
    port = int(os.environ.get("PORT", args.port))
    host = args.host
    uvicorn.run(app, host=host, port=port)
