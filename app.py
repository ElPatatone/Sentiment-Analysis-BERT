import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd


#Loading in the bert model for the sentiment analysis
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


#creating streamlit app
def main():

    #make the app look nicer
    st.set_page_config(page_title='sentiment analysis')

    hide_menu = """
                <style>
                
                footer {visibility: hidden; } 
                </style>
    """
    st.markdown(hide_menu, unsafe_allow_html=True)
    st.title('Yelp reviews Sentiment analysis')
    
    st.write('')
    st.write('')

    st.markdown("""
    ## What is this?
    This is an implementation of sentiment analysis using a pretrained BERT model from huggingface.
    The model has also been finetuned for product reviews, it will output a score from 1 - 5 for the given text input.

    #### How to use this?
    You can either unput your own text or copy and paste a link for a resturant on yelp and the program will scrape the page for the reviews.
    #### Take a look at the huggingface model used
    - [model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
    #### You can find the code for this streamlit app in here: 
    - [source code](https://github.com/ElPatatone/Sentiment-Analysis-BERT) 
    """)
    st.write('')
    st.write('')

    #analysing user input
    st.subheader('Enter your text')
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Enter Text Here")
        submit_button = st.form_submit_button(label='Analyze')

    if submit_button:
        #using the bert model to analyse text
        tokens = tokenizer.encode(raw_text, return_tensors='pt')
        result = model(tokens)
        score = int(torch.argmax(result.logits))+1
        st.info("The score is: {}".format(score))

    #analysing yelp reviews
    st.subheader('Paste in Yelp link')
    st.markdown("""
    To save on processing time, this will only output the first 10 reviews.
    """)
    with st.form(key='nlpforms'):
        link = st.text_input("link")
        submit_button2 = st.form_submit_button(label='Analyze')

    if submit_button2:
        #using the bert model to analyse the yelp reviews
        r = requests.get(link)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class':regex})
        reviews = [result.text for result in results]

        def sentiment_score(review):
            tokens = tokenizer.encode(review, return_tensors='pt')
            result = model(tokens)
            return int(torch.argmax(result.logits))+1

        df = pd.DataFrame(np.array(reviews), columns=['review'])
        df.index = np.arange(1, len(df) + 1)

        df['score'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

        st.dataframe(df)



if __name__ == '__main__':
    main()