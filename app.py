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
    st.title('Sentiment analysis on Yelp reviews ')
    
    st.write('')
    st.write('')

    st.markdown("""
    ## What is this?
    This is an implementation of sentiment analysis using a pretrained BERT model from huggingface.
    The model has also been finetuned for product reviews, it will output a score from 1 - 5 for the given text input. 

    **[model used](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)**  
    **[source code for project](https://github.com/ElPatatone/Sentiment-Analysis-BERT)**
    #### How to use this?
    Paste a link for a resturant on yelp and the program will scrape the page for the reviews. It will then run sentiment analysis on it and output a score 
    between 1 to 5 for the reviews, 1 being bad and 5 being good.  

    The program will then reuturn a pandas dataframe with the first 10 reviews for the given yelp site.
    Each review will have a score from 1 to 5 depending on the outcome of the sentiment analysis model.
   
  
    """)
    st.write('')
    st.write('')

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