#ライブラリの読み込み
import time
from scipy.fft import irfft
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler #スケーリングの為に追記
import joblib # モデル保存用
from io import BytesIO

#stremlitがリロードされない用にcallback関数を定義
def callback():
      st.session_state.key=True


#タイトル
st.title("機械学習アプリ_予測値の出力")


# 以下をサイドバーに表示
st.sidebar.markdown("### モデルファイルを入力してください")

#ファイルアップロード
uploaded_files = st.sidebar.file_uploader("Choose a model file", accept_multiple_files= False)

#ファイルがアップロードされたら以下が実行される
if uploaded_files:
    model = joblib.load(uploaded_files)
    

    #予測に使用する説明変数を入力する
    st.sidebar.markdown("### 予測に使用する説明変数のcsvファイルを入力してください")
    
    #ファイルアップロード
    uploaded_file2 = st.sidebar.file_uploader("Choose a csv faile", accept_multiple_files= False , key=2)

    #ファイルがアップロードされたら以下が実行される
    if uploaded_file2:
        df = pd.read_csv(uploaded_file2)

        y_pd_all=pd.DataFrame() #空のデータフレームを用意
        #予測値を行の数だけ計算する
        for i in range(len(df)):
            x=df.iloc[i,:]
            y=model.predict([x])
            y_pd = pd.DataFrame(y , columns=["pred"] , index=[i])
            y_pd_all = y_pd_all.append(y_pd)

        df_all = pd.concat([df , y_pd_all] , axis = 1)
        st.write("予測値の出力")
        st.write(df_all)
        # データフレームをcsv化
        csv = df_all.to_csv(encoding = "utf-8")
        # ダウンロードリンク作成
        st.download_button("Download csv" , data = csv , file_name="output.csv")
    