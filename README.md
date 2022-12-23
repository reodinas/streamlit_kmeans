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
* `K-Means Clustering.ipynb`: app.py 를 구현하기 위해 테스트한 코드
* `Kmeans_result.csv` : Kmeans Clustering 결과를 저장한 파일
* `requirements.txt`: 프로젝트에 사용한 라이브러리. 설치방법: `pip install -r requirements.txt`


# Usage
<http://ec2-3-39-253-47.ap-northeast-2.compute.amazonaws.com:8503/>

# Debugging
프로젝트 진행 중 겪은 에러와 해결방법은 티스토리에 정리해 두었습니다.

[<img src="https://img.shields.io/badge/Tistory-000000?style=for-the-badge&logo=Tistory&logoColor=white">](https://donghyeok90.tistory.com/category/Debugging)

# Improvement
csv파일을 업로드,저장 할때 파라미터(ex. index_col) 추가 기능 구현이 필요합니다.

Hierarchical Clustering도 구현했으나 AWS 프리티어의 사양 문제로 Dendrogram을 그리는데 너무 많은 시간이 소요되어 제외했습니다.

마찬가지로, 사양문제로 K-Means Clustering도 데이터가 많은 경우 처리시간이 오래 걸립니다.






