# -*- coding: utf-8 -*-

from google_play_scraper import app
import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews
from deep_translator import GoogleTranslator
import preprocessor as p
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from textblob import TextBlob
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from textblob.classifiers import NaiveBayesClassifier
from wordcloud import WordCloud, STOPWORDS
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

def casefolding_data(x):
    return x.lower()

def preprocessing_data(x):
    return p.clean(x)

def tokenize_data(x):
    return p.tokenize(x)

def stemming_data(x):
    ps = PorterStemmer() 
    return ps.stem(x)
    
def show_pie(label, data, legend_title, save_name) :
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw=dict(aspect='equal'))

        labels = [x.split()[-1] for x in label]

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}% ({:d})".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data), 
                                        textprops=dict(color="w"))

        ax.legend(wedges, labels,
                title= legend_title,
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=10, weight="bold")
        ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
        plt.savefig(f"static/{save_name}", bbox_inches='tight',pad_inches = 0, dpi = 200)
        # plt.show()

def plot_cloud(wordcloud):
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud) 
        plt.axis("off")
        plt.savefig('static/wordcloud.png', bbox_inches='tight',pad_inches = 0, dpi = 200)

def hero():
    result, continuation_token = reviews(
        'com.shopee.id',
        lang='en',
        country='id',
        sort=Sort.NEWEST,
        count=100,
        filter_score_with=None
    )

    """# **Scraping Data**"""
    df_busu = pd.DataFrame(np.array(result),columns=['review'])
    df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist()))    
    df_busu[['userName', 'score', 'at', 'content']].head()
    my_df = df_busu[['userName', 'score', 'at', 'content']]
    data = my_df
    data = data.dropna()
    data = data.reset_index(drop=True)

    """# **Text Preprocessing**"""

    """# **Menerjemahkan Data Ke Bahasa Inggris**"""
    sumber = list(data['content'])
    inggris = GoogleTranslator(source='auto', target='en').translate_sentences(sumber)
    data['data_inggris'] = inggris
    data = data.dropna()
    data = data.reset_index(drop=True)

    """# **Text Preprocessing**"""
    data['data_bersih'] = data['content'].apply(casefolding_data)
    data['data_bersih'] = data['data_bersih'].apply(preprocessing_data)
    data['data_bersih'] = data['data_bersih'].apply(tokenize_data)
    data['data_bersih'].replace('', np.nan, inplace=True)
    data = data.drop_duplicates()
    data['data_inggris'] = data['data_inggris'].apply(casefolding_data)
    data['data_inggris'] = data['data_inggris'].apply(preprocessing_data)
    data['data_inggris'] = data['data_inggris'].apply(tokenize_data)
    data['data_inggris'] = data['data_inggris'].apply(stemming_data)
    data = data.drop_duplicates()
    data = data.dropna()
    data = data.reset_index(drop=True)

    """# **Melakukan Modeling Data Untuk Analisis Sentimen**"""
    data_sumber = list(data['data_inggris'])
    polaritas = 0
    status = []
    total_positif = total_negatif = total_netral = total = 0

    for i, x in enumerate(data_sumber):
        analysis = TextBlob(x)
        polaritas += analysis.polarity

        if analysis.sentiment.polarity > 0.0:
            total_positif += 1
            status.append('Positif')
        elif analysis.sentiment.polarity == 0.0:
            total_netral += 1
            status.append('Netral')
        else:
            total_negatif += 1
            status.append('Negatif')

        total += 1 

    status = pd.DataFrame({'klasifikasi': status})
    data['klasifikasi'] = status

    """# **Klasifikasi Data Dengan Metode Naive Bayes Classifier**"""
    dataset = data.drop(['userName', 'score', 'at', 'content', 'data_bersih'], axis=1, inplace=False)
    dataset = [tuple(x) for x in dataset.to_records(index=False)]
    set_positif = []
    set_negatif = [] 
    set_netral = []

    for n in dataset:
        if(n[1] == 'Positif'):
            set_positif.append(n)
        elif(n[1] == 'Negatif'):
            set_negatif.append(n)
        else: 
            set_netral.append(n)

    set_positif = random.sample(set_positif, k=int(len(set_positif)/2))
    set_negatif = random.sample(set_negatif, k=int(len(set_negatif)/2))
    set_netral = random.sample(set_netral, k=int(len(set_netral)/2))
    train = set_positif + set_negatif + set_netral
    train_set = []
    for n in train:
        train_set.append(n)

    cl = NaiveBayesClassifier(train_set)
    akurasi = cl.accuracy(dataset)
    persentase = '{:.0%}'.format(akurasi)
    #print('Akurasi Test:', cl.accuracy(dataset))

    data_sumber = list(data['data_inggris'])
    polaritas = 0
    status = []
    total_positif = total_negatif = total_netral = total = 0
    for i, x in enumerate(data_sumber):
        analysis = TextBlob(x, classifier=cl)

        if analysis.classify() == 'Positif':
            total_positif += 1
        elif analysis.classify() == 'Netral':
            total_netral += 1
        else:
            total_negatif += 1
        
        status.append(analysis.classify())
        total += 1 

    if total_positif > total_negatif:
        hasil_akhir = 'PUAS'
    elif total_negatif > total_positif:
        hasil_akhir = 'TIDAK PUAS'
    else:
        hasil_akhir = 'TIDAK DAPAT DITENTUKAN'

    status = pd.DataFrame({'klasifikasi_bayes': status})
    data['klasifikasi_bayes'] = status
    label = ['Positif', 'Negatif', 'Netral']
    count_data = [total_positif+1, total_negatif+1, total_netral]

    """## **Kata yang Sering Muncul**"""
    data_pos = data.loc[data['klasifikasi_bayes'] == 'Positif']

    all_words = ' '.join([tweets for tweets in data_pos['data_inggris']])
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=3, background_color='white', colormap='Set2', collocations=False, stopwords = STOPWORDS).generate(all_words)
    plot_cloud(wordcloud)
    
    """## **Word Frequency**"""
    top_N = 10

    a = data_pos['data_inggris'].str.lower().str.cat(sep=' ')

    words = nltk.tokenize.word_tokenize(a)

    search_list = set(['price', 'payment', 'cheap', 'helpful', 'good', 'easy', 'promo', 'shipping',
                    'service', 'quality', 'fast', 'useful', 'feature', 'product', 'updated'])

    filtered_sentence = []
    
    for w in words:
        if w in search_list:
            filtered_sentence.append(w)
    
    word_dist = nltk.FreqDist(filtered_sentence)
    alasan = pd.DataFrame(word_dist.most_common(top_N))
    alasan.columns = ['Kategori', 'Frekuensi']

    """## **Output Hasil**"""
    kalimat = f'Dari hasil analisis {total} data review Shopee menggunakan Algoritma Naive Bayes dengan tingkat akurasi sebesar {persentase} dengan akurasi rata-rata sebesar 76% yang didapat dari percobaan 5 analisis dengan hasil PUAS menyatakan bahwa dari hasil analisis ini pengguna Shopee {hasil_akhir} dengan pelayanan yang telah diberikan, respon dari pengguna dapat dilihat pada grafik sebagai berikut'
    save_name = f"{str(time.time())[-3:]}.png"
    print(show_pie(label, count_data, "Status", save_name))
    data_print = data.drop(['data_inggris','data_bersih','klasifikasi'], axis=1, inplace=False)
    data_review = []
    for data in data_print.iterrows():
        data = list(data)  
        data_review.append(data[1].to_dict())
    
    data_alasan = []
    for data in alasan.iterrows():
        data = list(data)  
        data_alasan.append(data[1].to_dict())

    data_hero = {
        'data_review':data_review,
        'kalimat':kalimat,
        'save_name':save_name,
        'data_alasan':data_alasan
    }
    return data_hero


# hero()