import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder


def main():
    
    st.title('K-Means Clustering App!')
    st.subheader('csv파일을 올리고 K-Means Clustering 수행할 수 있습니다.')
    st.sidebar.image('https://user-images.githubusercontent.com/120348461/209294709-bfbdf8a9-03c5-406d-9c98-403f8de1685b.jpg')
    st.sidebar.image('https://user-images.githubusercontent.com/120348461/209294238-c56b2b27-af3c-42db-bd87-e64c08c698f5.gif')
    st.sidebar.image('https://user-images.githubusercontent.com/120348461/209293030-c4b901c6-1fc7-4e0d-ad5b-fea1e833d9c6.jpg')
    # csv 파일을 업로드 할 수 있다.
    file = st.file_uploader('csv파일을 업로드 하세요', type=['csv'])
        

    # 2. 업로드한 csv파일을 데이터프레임으로 읽고
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.dataframe(df)

        # csv파일 인코딩이 utf-8이 아닌 경우 처리  
        except UnicodeDecodeError:
            csv_encoding = st.text_input('인코딩 방식을 입력해주세요')
            df = pd.read_csv(file, encoding=csv_encoding)
            st.dataframe(df)

        
        # 결측값 처리
        df.dropna(inplace=True)

        st.header('')
        # 3. k-means 클러스터링을 하기위해 X로 사용할 컬럼을 설정할수있다
        X_col = st.multiselect('학습에 사용할 열을 모두 선택하세요.', df.columns)
        if X_col:    
            X = df[X_col]
            st.dataframe(X)

            # 문자열이 들어있으면 처리한 후에 화면에 보여준다.
            X_new = pd.DataFrame()

            for name in X.columns:
                # 각 컬럼 데이터를 가져온다
                data = X[name]

                # dropna 했을 때 비어있는 인덱스가 생기는걸 처리
                data.reset_index(inplace=True, drop=True)
                
                # 문자열인지 아닌지 나눠서 처리한다.
                if data.dtype == object:
                    
                    # 문자열이면, 갯수가 2개인지 아닌지 파악해서
                    # 2개이면 레이블인코딩, 그렇지 않으면 원핫인코딩 한다.
                    if data.nunique() <= 2:
                        # 레이블 인코딩
                        label_encoder = LabelEncoder()
                        X_new[name] = label_encoder.fit_transform(data)
                    else:
                        # 원핫 인코딩
                        col_names = sorted(data.unique())
                        # ct = ColumnTransformer(['encoder', OneHotEncoder(), [0]], remainder='passthrogh')
                        # X_new[col_names] = ct.fit_transform(data)
                        X_new[col_names] = pd.get_dummies(data.to_frame())
                
                else:
                    # 숫자 데이터 처리
                    X_new[name] = data
            # scaling
            scaler = MinMaxScaler()
            X_new = scaler.fit_transform(X_new)
            # st.text('Min-Max Scaling')
            # st.dataframe(X_new)
            
            st.header('')
            # 4. WCSS를 확인하기위한 그룹의 갯수를 정할수있다. 
            st.subheader('Elbow Method를 위한 클러스터링 갯수 선택')

            
            if X_new.shape[0] < 10:
                default_value = X_new.shape[0]
            else:
                default_value = 10

            max_number = st.slider('최대 클러스터 선택', 2, 15, value=default_value)

            wcss = []
            for k in np.arange(1, max_number+1):
                kmeans = KMeans(n_clusters=k, random_state=5)
                kmeans.fit(X_new)
                wcss.append(kmeans.inertia_)
            # st.write(wcss)
        
            # 5. 엘보우 메소트 차트를 화면에 표시
            x_range = np.arange(1, max_number+1)
            fig = px.line(x=x_range, y=wcss)
            fig.update_layout(
                title=dict(text='<b>The Elbow Method</b>'),
                xaxis_title = '<b>Number of Clusters</b>',
                yaxis_title='<b>WCSS</b>'
                )
            st.plotly_chart(fig)

            # 6. 실제로 그룹핑할 갯수 선택
            select_list = x_range[1:]
            k = st.selectbox('차트를 보고 최적의 클러스터 갯수를 선택하세요.', select_list)

            # 7.위에서 입력한 그룹의 갯수로 클러스터링
            kmeans = KMeans(n_clusters=k, random_state=10)
            y_pred = kmeans.fit_predict(X_new)
            st.header('')
            st.subheader('클러스터링 결과')
            df['Group'] = y_pred
            st.dataframe(df)

            df.to_csv('Kmeans_result.csv')

            st.header('')
            st.subheader('그룹이 추가된 데이터를 다운로드 받으실 수 있습니다.')

            # 8. 그룹이 추가된 데이터를 다시 다운로드 받을 수 있다.
            with open('Kmeans_result.csv', 'rb') as f:
                download = st.download_button('다운로드 받기', f, file_name='Kmeans_result.csv') 
            if download:
                st.success('다운로드가 완료됐습니다.')


if __name__ == '__main__':
    main()