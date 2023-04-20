import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from PIL import Image

import chardet
import joblib
import requests

from processing_function import *

from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings("ignore") 

st.set_page_config(
    page_title="Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load tfidf
@st.cache_data()
def TFIDF():
    tfidf_model = joblib.load('models/TFIDF.pkl')
    return tfidf_model

# Load model
@st.cache_data()
def load_model():
    model = joblib.load('models/Logistic_Regression_model.pkl')
    return model

tfidf_model = TFIDF()
model = load_model()



# GUI

with st.sidebar:  
    choice = option_menu('Menu',
                          ['Business Objective', 'Deploy Project', 'New Prediction', 'New Prediction using API'],
                          default_index=0)

# menu = ['Business Objective', 'Overview & Recommend', 'New Prediction']
# choice = st.sidebar.selectbox('Menu', menu, format_func=lambda x: x)
if choice == 'Business Objective':
    st.title("Sentiment Analysis")
    st.header('Business Objective')
    st.subheader("1. Sentiment Analysis")
    st.write('''
    - Sentiment analysis, also known as opinion analysis, emotion analysis, is the use of natural language processing, computational linguistics, and biometrics to identify, extract, quantify, and study subjective information and emotional states in a systematic way. 
    - Sentiment analysis is widely applied to various documents such as reviews and survey responses, social media, online media, and materials for applications ranging from marketing to customer relationship management and clinical medicine.
    - Sentiment analysis is the process of analyzing and evaluating a person's opinion about a particular object (whether the opinion is positive, negative, or neutral). This process can be done by using rule-based approaches, Machine Learning techniques, or a hybrid of both methods.
    - Sentiment analysis is widely used in practice, especially in marketing and promotional activities. Analyzing user reviews and feedback about a product to determine whether they are positive, negative, or neutral, or to identify the limitations of the product, can help companies improve product quality, enhance company's face, and increase customer satisfaction.
    ''')
    image1 = Image.open('pictures/sentiment_analysis.jpg')
    st.image(image1, caption='Sentiment Analysis')
    st.write("Source: https://vi.wikipedia.org/wiki/Ph%C3%A2n_t%C3%ADch_t%C3%ACnh_c%E1%BA%A3m")

    st.subheader("2. Sentiment Analysis in E-commerce")
    st.write('''
    - Nowadays, the demand for online shopping is increasing rapidly. Without having to go far, we can visit e-commerce websites to purchase almost everything.
    - To select a product, we tend to consider comments from those who have bought/tried the product to make a decision on whether to purchase it or not.
    - Customer feedback is very important, as it can help suppliers improve the quality of goods/services and their attitude in order to maintain their reputation and attract new customers.
    ''')

    st.subheader("3. Shopee")
    st.write('''
    - Shopee is an all-in-one e-commerce ecosystem that includes shopee.vn, a top-ranked e-commerce website in Vietnam and the Southeast Asia region.
    - We can visit shopee.vn to view product information, reviews, comments, and make order.
        ''')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        image2 = Image.open('pictures/shopee.png')
        st.image(image2)
        # st.write("Source: https://shopee.vn/")

        image3 = Image.open('pictures/shopee_2.png')
        st.image(image3, caption='Top 10 most visited e-commerce websites in Southeast Asia in Q2/2019')
        # st.write("Source: https://www.similarweb.com/")
    with col3:
        st.write("")

    st.subheader("4. Business Objective")
    st.write('''
    - Based on customer reviews, collected from the comments and reviews section on https://shopee.vn/..., build a prediction model to help sellers quickly understand feedback from customers about their products or services (positive, negative, or neutral).
    - This helps sellers understand the business situation and customer opinions, which in turn helps them improve their services and products.
    ''')


if choice == 'Deploy Project':
    st.title('Deploy Project')
    st.header('1. Data understanding')
    st.write('''
    - The data is provided in the file "Products_Shopee_comments.csv"
    - The data has these columns as follows:
        - product_id
        - category
        - sub_category
        - user
        - rating
        - comment
    ''')
    st.header('2. Data preprocessing')
    st.write('#### The first 5 rows of dataset')
    image = Image.open('pictures/data_raw.png')
    st.image(image)
    
    st.write("#### Keep 2 columns:'rating' and 'comment' for prediction, drop others")
    st.code("df_new = df[['rating','comment']]")
    st.write('''
    #### Cleaning the text in column 'comment' by these steps as follows:
        - Convert emoji, teencode to words
        - Delete puchtuation, numbers and wrong words
        - Delete excess blank space
        - Standardize Vietnamese Unicode.
        - Using the underthesea library, keep only these following type of words as follows:
            - 'A': adjective
            - 'AB': adverb
            - 'V': verb
            - 'VB': special verb
            - 'VY': weak verb
            - 'R': preposition
        - Delete stopwords
    ''')
    st.info('For more details, please download the processing file below')
    with open('processing_function.py', 'rb') as f:
        data = f.read()
        encoding = chardet.detect(data)['encoding']
    with open('processing_function.py', 'r', encoding=encoding) as f:
        python_code = f.read()
    st.download_button(
        label='Download',
        data=python_code,
        file_name='function.py',
        mime='text/plain'
    )

    st.write("#### Map the rating from 1-5 to 2 groups: negative (0) and positive (1) and delete the neural rating (rating = 3)")
    st.code('''
    group = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    df['group'] = df['rating'].map(group)
    df_new = df[df['rating'] != 3]
    ''')
    st.write('#### Using wordcloud to find the keyword in each group')
    image = Image.open('pictures/wordcloud.png')
    st.image(image)

    st.write('#### Split the data into train and test sets')
    st.code('''
    X = df_final["comment_clean"]
    y = df_final["group"]
    ''')
    st.write('#### Using TF-IDF')
    st.code('''
    tfidf = TfidfVectorizer(max_features= 1500, min_df= 2)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)
    ''')
    st.write('#### Handle the imbalanced data by oversampling')
    st.code('''
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    X_test, y_test = sm.fit_resample(X_test, y_test)
    ''')

    st.header('3. Model Selection')
    st.write('''
    #### Run 3 model includes
        - Logistic Regression
        - Naive Bayes
        - Random Forest
             ''')
    
    st.write('#### Confusion matrix of each model')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("##### Logistic Regression")
        st.image("pictures/cls_report_lr.png")
        st.image("pictures/cfm_lr.png")

    with col2:
        st.write("##### Naive Bayes")
        st.image("pictures/cls_report_nb.png")
        st.image("pictures/cfm_nb.png")

    with col3:
        st.write("##### Random Forest")
        st.image("pictures/cls_report_rf.png")
        st.image("pictures/cfm_rf.png")

    st.write('#### Table comparing the results of the models')
    st.image('pictures/comparison_df.png')

    st.header('4. Conclusion')
    st.info('''
    Choose the **Logistic Regression** model because:\n
    - Accuracy score is quite high (nearly 87%)\n
    - Recall score is the highest among the selected models (nearly 90%)\n
    - Training time is short (less than 10s)
    
    ''')

if choice == 'New Prediction':
    st.title('New Prediction')
    lines = None
    type = st.selectbox("Select the data you'd like to predict:", options=("Input text", "Upload file *.csv","Upload file *.txt", ))
    if type == "Upload file *.csv":
        st.write("Please upload file with this format")
        frame = pd.DataFrame({'comment': ['Input your comment here']})
        st.dataframe(frame)
        uploaded_file = st.file_uploader("Choose file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            df = df['comment']
            lines = process_final(df)
            x_new = tfidf_model.transform(lines)
            y_pred_new = model.predict(x_new)
            result_y = pd.DataFrame(y_pred_new, columns=['Result']) 
            result = pd.concat([df, result_y], axis = 1)
            result['Result'] = result['Result'].map(lambda x: 'Negative' if x == 0 else 'Positive')
            st.write("Result:")
            st.dataframe(result)

            # Create a download button for the DataFrame
            button = st.download_button(
                label="Download CSV",
                data=result.to_csv(index=False, encoding='UTF-8'),
                file_name='result.csv',
                mime='text/csv;charset=UTF-8',
            )

    if type=="Input text":
        review = st.text_input(label='Input your feedback:')
        if st.button(label='Submit'):
            frame = pd.DataFrame({'comment': [review]})
            df = frame['comment']
            lines = process_final(df)
            x_new = tfidf_model.transform(lines)
            y_pred_new = model.predict(x_new)
            if y_pred_new == 1:
                st.info('Your comment is: **Positive**')
                pos = Image.open('pictures/positive.jpg')
                pos = pos.resize((400,400))
                st.image(pos, width=250)
            else:
                st.error('Your comment is: **Negative**')
                neg = Image.open('pictures/negative.png')
                neg = neg.resize((400,400))
                st.image(neg, width=250)

    if type == "Upload file *.txt":
        uploaded_file = st.file_uploader("Choose file", type='txt')
        if uploaded_file is not None:
            lines = uploaded_file.readlines()
            # Decode bytes to string
            lines = [line.decode("utf-8") for line in lines]

            frame = pd.DataFrame({'comment': lines})
            df = frame['comment']
            lines = process_final(df)
            x_new = tfidf_model.transform(lines)
            y_pred_new = model.predict(x_new)
            result_y = pd.DataFrame(y_pred_new, columns=['Result']) 
            result = pd.concat([df, result_y], axis = 1)
            result['Result'] = result['Result'].map(lambda x: 'Negative' if x == 0 else 'Positive')
            st.write("Result:")
            st.dataframe(result)

            # Create a download button for the DataFrame
            button = st.download_button(
                label="Download CSV",
                data=result.to_csv(index=False, encoding='UTF-8'),
                file_name='result.csv',
                mime='text/csv;charset=UTF-8',
            )
if choice == 'New Prediction using API':
    st.title('Sentiment Analysis using MonkeyLearn API')
    review = st.text_input(label='Input your feedback:')
    if st.button(label='Submit'):
        # Open the file in read mode
        with open('api_key.txt', 'r') as file:
            # Read the contents of the file
            key = file.read().strip()

        model_id = 'cl_pi3C7JiL'
        api_key = key
        url = f'https://api.monkeylearn.com/v3/classifiers/{model_id}/classify/'

        data = {
            'data': [
                {'text': review}
            ]
        }

        headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=data, headers=headers)

        sentiment_category = response.json()[0]['classifications'][0]['tag_name']
        if sentiment_category == 'Positive':
            st.info('Your comment is: **Positive**')
            pos = Image.open('pictures/positive.jpg')
            pos = pos.resize((400,400))
            st.image(pos, width=250)
        else:
            st.error('Your comment is: **Negative**')
            neg = Image.open('pictures/negative.png')
            neg = neg.resize((400,400))
            st.image(neg, width=250)
    
