import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

def main():
    st.title('K-Means 클러스터링 앱')

    # 1. csv 파일을 업로드 할 수 있다.
    file = st.file_uploader('csv파일을 업로드 하세요', type=['csv'])

    # 2. 업로드한 csv파일을 데이터프레임으로 읽고
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df)
        
        # 결측값 처리
        df.dropna(inplace=True)

        # 3. k-means 클러스터링을 하기위해 X로 사용할 컬럼을 설정할수있다
        X_col = st.multiselect('X로 사용할 컬럼을 선택하세요.', df.columns)
        if X_col:    
            X = df[X_col]
            st.dataframe(X)

            # 문자열이 들어있으면 처리한 후에 화면에 보여준다.
            X_new = pd.DataFrame()

            for name in X.columns:
                # 각 컬럼 데이터를 가져온다
                data = X[name]
                
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
                        ct = ColumnTransformer([ ('encoder', OneHotEncoder(), [0]) ],
                                remainder='passthrough')
                        col_names = sorted(data.unique())
                        X_new[col_names] = ct.fit_transform(data.to_frame())
                
                else:
                    # 숫자 데이터 처리
                    X_new[name] = data
            # scaling
            scaler = MinMaxScaler()
            X_new = scaler.fit_transform(X_new)
            st.dataframe(X_new)

            # 4. WCSS를 확인하기위한 그룹의 갯수를 정할수있다. 
            st.subheader('WCSS를 위한 클러스터링 갯수 선택')
            max_number = st.slider('최대 그룹 선택', 2, 15, value=10)

            wcss = []
            for k in np.arange(1, max_number+1):
                kmeans = KMeans(n_clusters=k, random_state=5)
                kmeans.fit(X_new)
                wcss.append(kmeans.inertia_)
            # st.write(wcss)
        
            # 5. 엘보우 메소트 차트를 화면에 표시
            fig1 = plt.figure()
            x = np.arange(1, max_number+1)
            plt.plot(x, wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            st.pyplot(fig1)

            # 6. 실제로 그룹핑할 갯수 선택
            k = st.number_input('그룹 갯수 결정', 2, max_number)

            # 7.위에서 입력한 그룹의 갯수로 클러스터링 하여 결과를 보여준다
            kmeans = KMeans(n_clusters=k, random_state=5)
            y_pred = kmeans.fit_predict(X_new)

            df['Group'] = y_pred
            st.dataframe(df.sort_values('Group'))

            # df.to_csv('result.csv')


if __name__ == '__main__':
    main()