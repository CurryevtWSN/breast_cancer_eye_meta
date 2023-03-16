#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)
#%%set title
st.set_page_config(page_title='Prediction model for ocular metastasis in breast cancer：Machine Learning-Based development and interpretation study')
st.title('Prediction model for ocular metastasis in breast cancer：Machine Learning-Based development and interpretation study')
st.sidebar.markdown('## Variables')
CEA = st.sidebar.slider("CEA(μg/L)", 0.00, 1000.00, value=400.00, step=0.01)
CA125 = st.sidebar.slider("CA125(μg/L)", 0.00, 1000.00, value=800.00, step=0.01)
CA153 = st.sidebar.slider("CA153(μg/L)", 0.00, 500.00, value=200.00, step=0.01)
Hb = st.sidebar.slider("Hb(g/L)",0, 210, value=130, step=1)
ALP = st.sidebar.slider("ALP(U/L)",0,500,value = 200, step = 1)
Ca = st.sidebar.slider("Ca(mmol/L)",0.00, 13.00, value = 8.00, step = 0.01)
LDL = st.sidebar.slider("LAL(mmol/L)",0.00, 18.00, value = 5.00, step = 0.01)
ApoA = st.sidebar.slider("ApoA(mmol/L)",0.00, 4.00, value = 2.00, step = 0.01)
TC = st.sidebar.slider("TC(mmol/L)",0.00,20.00, value = 4.00, step = 0.01)
CA199 = st.sidebar.slider("CA199(μg/L)", 0.00, 1000.00, value=500.00, step=0.01)
Histopathology = st.sidebar.selectbox('Histopathological_type',('Unkown','Infiltrative ductal carcinoma','Other Types'),index = 2)
Axillary_lymph_node_metastasis = st.sidebar.selectbox('Axillary lymph node metastasis',('Unkown','No','≤4','>4'),index = 3)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Unkown':1,'Infiltrative ductal carcinoma':2,'Other Types':3,'No':2,'≤4':3,'>4':4}

Histopathology =map[Histopathology]
Axillary_lymph_node_metastasis = map[Axillary_lymph_node_metastasis]

#%%load model
bag_model = joblib.load('bag_model.pkl')
BAG_model = bag_model
#%%load data
hp_train = pd.read_csv('breast_cancer.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["CEA","CA125","CA153","Hb","ALP",'Ca',"LDL",'ApoA','TC','Histopathology','CA199','Axillary_lymph_node_metastasis']
target = 'M'
y = np.array(hp_train[target])
sp = 0.5
#figure
is_t = (BAG_model.predict_proba(np.array([[CEA,CA125,CA153,Hb,ALP,Ca,LDL,ApoA,TC,Histopathology,CA199,Axillary_lymph_node_metastasis]]))[0][1])> sp
prob = (BAG_model.predict_proba(np.array([[CEA,CA125,CA153,Hb,ALP,Ca,LDL,ApoA,TC,Histopathology,CA199,Axillary_lymph_node_metastasis]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')
    # #%%cbind users data
    # col_names = features
    # X_last = pd.DataFrame(np.array([[CEA,CA125,CA153,Hb,ALP,Ca,LDL,ApoA,TC,Histopathology,CA199,Axillary_lymph_node_metastasis]]))
    # X_last.columns = col_names
    # X_raw = hp_train[features]
    # X = pd.concat([X_raw,X_last],ignore_index=True)
    # if is_t:
    #     y_last = 1
    # else:
    #     y_last = 0
    
    # y_raw = (np.array(hp_train[target]))
    # y = np.append(y_raw,y_last)
    # y = pd.DataFrame(y)
    # model = BAG_model
    # #%%calculate shap values
    # sns.set()
    # explainer = shap.Explainer(BAG_model, X)
    # shap_values = explainer.shap_values(X)
    # a = len(X)-1
    # #%%SHAP Force logit plot
    # st.subheader('SHAP Force logit plot of BAG model')
    # fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    # force_plot = shap.force_plot(explainer.expected_value,
    #                 shap_values[a, :], 
    #                 X.iloc[a, :], 
    #                 figsize=(25, 3),
    #                 # link = "logit",
    #                 matplotlib=True,
    #                 out_names = "Output value")
    # st.pyplot(force_plot)
    # #%%SHAP Water PLOT
    # st.subheader('SHAP Water plot of BAG model')
    # shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    # fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    # waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    # st.pyplot(waterfall_plot)
    # #%%ConfusionMatrix 
    # st.subheader('Confusion Matrix of BAG model')
    # bag_prob = bag_model.predict(X)
    # cm = confusion_matrix(y, bag_prob)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOM', 'OM'])
    # sns.set_style("white")
    # disp.plot(cmap='RdPu')
    # plt.title("Confusion Matrix of BAG")
    # disp1 = plt.show()
    # st.pyplot(disp1)
