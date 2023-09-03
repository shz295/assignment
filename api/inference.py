from keras.models import load_model
from preprocessing import preprocess
import numpy as np
from cv2 import imwrite

# 저장된 모델 불러오기
model = load_model('lenet_os.keras')


def get_class(i):
    classes = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)] + ['free']
    return classes[i]

def get_category(i):
    return '숫자인식' if i < 10 else '알파벳인식' if i < 36 else '자유패턴'

# 이미지 추론
def infer(image, return_r=False):
    # 이미지 전처리
    main = preprocess(image, img_size=(28, 28), r=False)
    gray = preprocess(image, img_size=(28, 28), r=True)

    # 추론
    main_pred = model.predict(main.reshape(1, 28, 28, 1), verbose=0)[0]
    gray_pred = model.predict(gray.reshape(1, 28, 28, 1), verbose=0)[0]

    main_pred_idx = np.argmax(main_pred[:36])
    gray_pred_idx = np.argmax(gray_pred[:36])

    pred, r, confidence = None, None, None

    if main_pred[36] > 0.5 and gray_pred[36] > 0.5:
        pred = 36
        confidence = max(main_pred[36], gray_pred[36])
        r = False

    elif main_pred[main_pred_idx] > gray_pred[gray_pred_idx]:
        pred = main_pred_idx
        confidence = main_pred[main_pred_idx]
        r = False

    else:
        pred = gray_pred_idx
        confidence = gray_pred[gray_pred_idx]
        r = True
    
    return (pred, r, confidence) if return_r else (pred, confidence)