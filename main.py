import streamlit as st
import pandas as pd
import json
import gensim.downloader as api

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

from re import sub

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

@st.cache
def get_data(filename):
    data = []
    for line in open(filename, 'r'):
        data.append(json.loads(line))
    MM_DATA = pd.DataFrame.from_records(data)
    return MM_DATA


skills_dataset = get_data('data/mm.json')

with header:
    st.title('Semantic Similarity Recommender System For MM Dataset')
    st.text("")
    st.text("")
    st.text("")
    st.text('In this project I build a semantic similarity recommender system using MM Data')

with dataset:
    st.header('MM Raw Dataset')
    st.text("")
    st.text("")
    st.text("")
    st.text('I got this dataset from JSON file with MM Data')

    st.write(skills_dataset.head())

    st.subheader('Mentors')
    df_mentor = skills_dataset.loc[skills_dataset['mode'] != 'mentee']
    st.write(df_mentor)

    st.subheader('Mentees')
    df_mentee = skills_dataset.loc[skills_dataset['mode'] != 'mentor']
    st.write(df_mentee)



with features:
    st.header('Features')

    st.markdown('* **Input:** I created this feature in order combine all the relevant parameters required for input')

with model_training:
    st.header('Training')
    st.text('Here you can get recommendation for any individual')

    # Insert record sample
    uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

    if uploaded_file:
        sample_df = pd.read_excel(uploaded_file)

        st.dataframe(sample_df)


    sel_col, disp_col = st.beta_columns(2)


    
    input_name = sel_col.text_input("Input Name from Dataset here", "kamaté")
    sel_col.subheader('Name Details')
    
    if 'sample_df' in locals():
        skills_dataset = skills_dataset.append(sample_df)
    else:
        pass

    sel_col.write(skills_dataset.loc[skills_dataset['name'] == input_name])


    df = skills_dataset.copy()
    del df['last_login']
    del df['build_number']
    del df['mana']
    del df['updatedAt']
    del df['lastMessageTime']
    del df['founder']
    del df['admin']
    del df['blocked']

    df = df[['name', 'industriesMentee', 'aoiMentee', 'language', 'aoiMentor', 'industriesMentor', 
                  'why', 'expectationsFromMentor', 'whyBecomeMentee', 
                  'what', 'mentoring', 'location', 'job', 'mode']]

    df = df.dropna()

    # converts the following columns into string format
    list_inputs = ['name', 'industriesMentee', 'aoiMentee', 'language', 'aoiMentor', 'industriesMentor', 
                    'why', 'expectationsFromMentor', 'whyBecomeMentee', 
                    'what', 'mentoring', 'location', 'job', 'mode']
    for i in list_inputs:
        df[i] = df[i].astype(str)

    # removes remove leading and trailing whitespaces in Python strings
    for i in df['name']:
        df['name'] = df['name'].replace([i],i.strip())

    # removing all rows with 'NA', except for the 'job' row
    list_inputs_NA = ['name', 'industriesMentee', 'aoiMentee', 'language', 'aoiMentor', 'industriesMentor', 
                    'why', 'expectationsFromMentor', 'whyBecomeMentee', 
                    'what', 'mentoring', 'location', 'mode']
    for i in df.columns:
        df = df[df[i] != 'NA']


    df['Input'] = df[['industriesMentee', 'aoiMentee', 'language', 'aoiMentor', 'industriesMentor', 
                  'why', 'expectationsFromMentor', 'whyBecomeMentee', 
                  'what', 'mentoring', 'location', 'job']].apply(lambda x: ' '.join(x), axis = 1)

    metadata = df.copy()
    df_mentor = metadata.loc[metadata['mode'] != 'mentee']
    df_mentor_list = list(df_mentor['name'])

    stopwords = ['the', 'and', 'are', 'a']

    def preprocess(doc):
        # Tokenize, clean up input document string
        doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        doc = sub(r'<[^<>]+(>|$)', " ", doc)
        doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
        doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

    corpus = [preprocess(document) for document in metadata['Input']]


    # Load the model: this is a big file, can take a while to download and open
    glove = api.load("glove-wiki-gigaword-50")    
    similarity_index = WordEmbeddingSimilarityIndex(glove)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus)
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.  
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in corpus]],
            similarity_matrix)

    cosine_model=index
    cosine_model_df=pd.DataFrame(cosine_model,index=df.name,columns=df.name)

    def make_recommendations(employee_likes):
        employee_likes = str(employee_likes)
        recommender = cosine_model_df[employee_likes].sort_values(ascending=False)[:10]
        recommender = recommender.to_frame()
        recommender.reset_index(level=0, inplace=True)
        recommender = recommender.rename(columns={employee_likes: "Soft Cosine Similarity"})
        recommender = recommender.loc[recommender['name'] != employee_likes]
        recommender = recommender.loc[recommender['name'].isin(df_mentor_list)]
        recommender = recommender.reset_index(drop=True)
        return recommender

    recommender = make_recommendations(input_name)

    disp_col.subheader('Recommendations')
    recommender.reset_index(level=0, inplace=True)
    recommender = recommender.rename(columns={input_name: "Soft Cosine Similarity"})
    disp_col.write(recommender)

    disp_col.subheader('Recommendations Details')
    for i in recommender.name:
        disp_col.write(df.loc[df['name'] == i])

    


