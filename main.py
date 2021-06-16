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
df_photo = skills_dataset.copy()
df_photo = df_photo[['name', 'imageUrl']]


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


    sel_col, disp_col, pic_col = st.beta_columns(3)


    
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
        recommender = pd.merge(recommender, df_photo, how='left', left_on='name', right_on='name')
        recommender["imageUrl"].fillna("data:image/jpeg;base64,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", inplace = True)
        return recommender

    recommender = make_recommendations(input_name)

    disp_col.subheader('Recommendations')
    recommender.reset_index(level=0, inplace=True)
    recommender = recommender.rename(columns={input_name: "Soft Cosine Similarity"})
    disp_col.write(recommender)

    # disp_col.subheader('Recommendations Details')
    # for i in recommender.name:
    #     disp_col.write(df.loc[df['name'] == i])

    pic_col.subheader("Recommendations Photo")
    for j in recommender.imageUrl:
        pic_col.image(j, width = 35)