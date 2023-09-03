# 사용법
docker를 통한 실행
```
docker compose build
docker compose up
```

필요한 패키지를 설치
```
pip install -r requirements.txt
```

API 실행법
```
python main.py
```

인터페이스 실행법
```
streamlit run interface.py
```

실행 후 http://localhost:8501/ 에 인터페이스 접속 가능하며 http://localhost:8000/docs 에서 API 문서 확인 가능

# 접근 방법 / 시도

데이터 탐색, 전처리, 모델링 등의 과정은 exploration.ipynb 파일에 적혀있습니다.

탐사 과정은 다음과 같았습니다: 먼저, 데이터셋의 레이블링 시스템과 디렉토리를 이해하려고 합니다. 데이터를 불러온 후, 데이터를 훈련 및 테스트 데이터셋으로 분할합니다. 다음으로, 데이터가 올바르게 처리되었는지 확인하기 위해 일부 이미지를 살펴봅니다. 그 다음, 이미지를 244x244 크기로 조정하고 정규화된 픽셀로 변환한 이미지에 대해 간단한 모델로 기준선 정확도를 설정합니다. 

정확도를 향상시키는 여러 가지 방법을 탐구합니다. 모델이 인식하기 쉽도록 이미지를 개선하는 방법을 찾기 위해 OpenCV를 사용합니다. 이 과정에서 이미지를 보드 크기로 왜곡시키는 작업이 수행되었습니다. 배경 제거도 시도되었지만 아직 사용하기에 더 많은 작업이 필요하다고 결정되었습니다. 이미지 전처리 방법을 정리하고, 작성한 함수로 이루어진 파일인 preprocessing.py를 작성했습니다. 

그 후에는 ResNet50, VGG16, MobileNetV2 전이학습된 모델을 시도해봤습니다. ResNet와 VGG16은 이런 간단하고 작은 데이터량의 경우에 적합하지 않다고 판단했습니다. 설령 데이터 증강을 사용하더라도 파인튜닝 해봤자 아무 차이 보이지 않습니다. 그런데, 숫자 분류의 제을 좋은 결과는 MobileNetV2를 통해 이루었습니다. 데이터 크기를 증가시키기 위해 데이터 증강을 시도하고, 숫자, 문자, 자유 패턴의 개수 차이를 고려하기 위해 클래스 가중치를 사용합니다. 최종적으로 숫자 분류 모델에 데이터 증강을 적용했습니다. 이러한 탐구는 model_selection_2.ipynb에서 확인이 가능합니다. 정확도를 향상시키는 방법을 탐구한 후, 하이퍼파라미터 최적화를 위해 model_generation.ipynb이라는 새로운 파일을 만들었습니다. 먼저 GridSearchCV를 사용하여 다양한 매개변수 조합을 시도했지만, 일관적으로 RAM이 부족하여 매개변수 조합을 무작위로 시도하고 어느 정도 잘 하는 것을 선택하기로 결정했습니다. 이 방법을 통한 최종 모델 선택은 model_selection.ipynb에 설명되어 있습니다.

그러나, 'model_selection.ipynb'에서 하이퍼파라미터를 찾던 중 데이터 누출이 발견되었습니다. 훨씬 작은 데이터셋과 더 복잡한 작업에서 최상위 MNIST 모델과 유사한 점수를 받는 것은 이상했으며, 사용하지 않은 폴더에서 샘플 외 예측을 수행한 후 이것이 확인되었습니다. 실제로 모델들은 데이터셋 밖에서는 무작위 추측과 유사한 예측 능력을 가지고 있었습니다. 하나의 가설은 색상을 반전한 동일한 이미지들이 훈련 및 검증/테스트 세트 양쪽에 골고루 분포되어 있어 모델이 알지 말아야 할 데이터를 학습하게 된 것일 수 있습니다. 확실히 이것은 데이터 누출의 원인이 될 수 있지만, 'r'이 아닌 이미지만 로드하여 이를 제어한 후에도 차이는 그다지 크지 않았습니다. 또한 다른 고려 사항으로는 다른 파일명을 가진 중복 이미지가 있는지 여부였지만, 중복 파일명이 존재하지 않는지 SSIM 또는 데이터 로딩 방법을 비교해본 결과 아무 변화가 없었습니다. 오염된 테스트 세트의 처리나 클래스 계층화의 차이도 유의미한 결과를 보여주지 않았습니다. 이미지 전처리를 검토한 결과 두 경우 모두 동일한 이미지가 보였습니다. 결국, 이상적이지는 않지만 '추가' 디렉토리를 테스트/검증용으로 사용하기로 결정되었으며, '자유패턴'에서 일치하는 클래스 계층화를 가진 몇 개의 무작위 디렉토리도 선택되었습니다. '원샷' 스타일 모델의 가능성도 탐구되었는데, 직접 37개 카테고리 중 하나를 선택하는 것이 처음에는 3개 카테고리의 모델을 선택한 다음 해당 문자/숫자를 갖는 다른 모델을 선택하는 것과 유사한 결과를 제공할지 여부를 확인하기 위한 것이었습니다. 결국 이 모델은 두 작업 모두에서 성능이 떨어졌습니다.

이러한 모델을 사용하여 FastAPI를 사용하여 간단한 RESTful API를 생성했습니다. 이 API는 Uvicorn을 통해 웹 서버로 제공됩니다. Uvicorn은 웹 서버를 8000 포트에 바인딩하고, FastAPI는 /predict/ 경로에 post 메서드로 엔드포인트를 제공합니다. 다른 포트 사용할 경우에는 --port으로 설정이 가능합니다. 파일이 전송되면 OpenCV의 imdecode를 사용하여 전송된 바이트를 예측을 위한 numpy 배열로 변환합니다. 추론 파이프라인 자체는 inference.py에 infer() 메인 함수로 설명되어 있습니다. 특정 작업을 위해 이미지 전처리가 적용되고, 예측이 수행되며 결과와 신뢰도가 반환됩니다. 카테고리(숫자/알파벳/자유 패턴)의 예측이 이루어지고, 결과에 따라 숫자/알파벳의 예측이 이루어집니다. 마지막으로, 예측된 카테고리와 경우에 따라 자유 패턴이 아닌 특정 숫자/문자가 포함된 사전 객체가 반환됩니다.

그런 다음 streamlit을 사용하여 API와 상호 작용하는 간단한 웹 인터페이스를 만들었습니다. 이 인터페이스를 통해 이미지 파일을 업로드하고 업로드한 이미지를 표시할 수 있습니다. 적절한 이미지가 업로드되면 예측 버튼이 표시되며, 누르면 이미지가 http://localhost:8000/predict로 전송됩니다. 선택적으로 --api_address 커맨드 라인 인수를 사용하여 API의 위치를 지정할 수 있습니다(로컬로 호스팅되지 않은 경우). 포트는 --port으로 선택도 가능합니다. 결과를 받은 후에 예측된 카테고리, 특정 숫자 또는 알파벳 문자, 그리고 신뢰도 수준이 사용자에게 제시됩니다.

# 평가

테스트 세트의 최종 평가는 다음과 같습니다.
| 모델                    | 범주형 교차 엔트로피 오차 | 정확도  |
|-------------------------|-----------------------|--------|
| 숫자/알파벳/자유패턴 인식  | 0.78073               | 72.658 |
| 숫자 인식                | 2.12553               | 33.000 |
| 알파벳 인식              | 3.529                 | 27.308 |

# 결론

솔직히는 모델들의 예측 능력이 매우 불만족합니다. 이런 과제에는 모델이나 하이퍼파라미터 선택보다 이미지 전처리가 제일 중요하고 둘째는 데이터 증강인 것 같습니다. 전처리 잘하면 MNIST급 문제 될 수도 있고 데이터 증강 잘하면 데이터량 문제도 줄이니 90%의 정확도가 가능하지 않을까 합니다. 