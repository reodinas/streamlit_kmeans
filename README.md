# Preview
<img src="https://user-images.githubusercontent.com/120348461/209302353-51c00d6d-69f1-40a6-a511-4232501d4c03.jpg">
<img src="https://user-images.githubusercontent.com/120348461/209302360-a7cd09cf-9baf-4638-8b90-011309597cad.jpg">


# Overview
* 유저가 csv 파일을 올리고 K-Means Clustering을 수행할 수 있는 앱입니다.
* 학습에 사용할 컬럼과 k값을 입력 받습니다.
* 결측값을 처리합니다.
* 선택된 컬럼 중에 `dtype`이 숫자인 컬럼은 그대로 가져옵니다.
* 선택된 컬럼 중에 `dtype`이 `object`인 컬럼은 범주가 2개면 Label 인코딩, 3개 이상이면 One-Hot 인코딩을 수행합니다.
* Elbow Method를 사용해 그래프를 보여주고, 유저는 k값을 선택합니다.
* 선택한 k값으로 K-Means Clustering을 하고 그룹 정보가 담긴 컬럼을 추가해 표시합니다.
* 유저는 그룹 정보가 추가된 데이터를 다운로드 받을 수 있습니다.
* AWS EC2 프리티어 서버를 사용했습니다.
* GitHub Actions를 사용하여 CI/CD 합니다.

# Stack
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"><img src="https://img.shields.io/badge/Amazon EC2-FF9900?style=for-the-badge&logo=Amazon EC2&logoColor=white"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"><img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white">



# Files
* `app.py`: main 실행 파일
* `app.home.py`: Home 페이지 모듈
* `app_EDA.py`: Chart 페이지 모듈
* `app_admin.py`: admin 페이지 모듈
* `my_function.py`: app_EDA에서 사용되는 차트들을 정의한 모듈
* `df_origin.csv`: 앱에서 사용, 저장되는 데이터
* `df_origin_backup.csv` : df_origin.csv의 백업파일
* `requirements.txt`: 프로젝트에 사용한 라이브러리. 설치방법: `pip install -r requirements.txt`
* `data` 폴더: df_origin.csv를 만들 때 사용했던 csv 파일들. 참고용으로 넣어놨지만 앱의 동작과 아무 관련 없습니다.
* `데이터확인.ipynb`: 데이터를 파악하기 위해 테스트한 코드
* `lsr1.ipynb`: 데이터프레임을 만들 때 테스트한 코드
* `lsr_eda.ipynb`: 차트를 만들 때 테스트한 코드
* `Classification.ipynb`: 머신러닝을 할 때 테스트한 코드

# Usage
<http://ec2-3-39-253-47.ap-northeast-2.compute.amazonaws.com:8502/>
<br>admin 페이지 비밀번호 : abc123
# Debugging
프로젝트 진행 중 겪은 에러와 해결방법은 티스토리에 정리해 두었습니다.

[<img src="https://img.shields.io/badge/Tistory-000000?style=for-the-badge&logo=Tistory&logoColor=white">](https://donghyeok90.tistory.com/category/Debugging)

# Improvement

이 데이터는 Null 값은 없지만 가구소득정보에 응답하지 않은 데이터는 무응답이라고 저장되어 있습니다.

그리고 정보에 가명처리를 해서인지, 가구소득정보는 수치가 아니라 범위로 구분되어 있습니다.

그래서 머신러닝 분류모델을 이용해 무응답한 사람들의 소득정도를 예측하려 했습니다.

하지만 상관계수를 확인해 본 결과 상관관계가 거의 보이지 않았고,

Scikit-learn의 DecisionTree, LogisticRegression, SVC, RandomForest 모델들을 사용해 머신러닝한 결과, 정확도가 모두 30% 대에 그쳤습니다.

딥러닝 알고리즘을 사용해 볼 수도 있겠지만. AWS 프리티어의 사양으로 딥러닝이 탑재된 앱을 배포하기엔 무리라고 판단했습니다.

추후에 기회가 된다면 딥러닝으로 다시 학습시켜보도록 하겠습니다.







