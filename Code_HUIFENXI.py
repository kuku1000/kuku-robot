# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tensorflow import metrics
from wordcloud import WordCloud  #词云库
from PIL import Image
import numpy as np
import jieba  #用于中文分词的库
import jieba.posseg as pseg
import jieba.analyse

font = "C:/Windows/Fonts/msjhbd.ttc"
# text_from_file_with_apath = ""
text_from_file_with_apath = open('C:/Users/10513/Desktop/数据/AnalysisReviews1/AnalysisReviews/review_client.txt','r',encoding='utf8').read()
print(text_from_file_with_apath)

text_from_file_with_apath = text_from_file_with_apath.replace('我们', '')
text_from_file_with_apath = text_from_file_with_apath.replace('的', '')
text_from_file_with_apath = text_from_file_with_apath.replace('问题', '')
text_from_file_with_apath = text_from_file_with_apath.replace('页面', '')
# wordlist_after_jieba = jieba.cut(text_from_file_with_apath)
# jieba分词 基于 TF-IDF 算法的关键词抽取
wordlist_after_jieba_grade = jieba.analyse.extract_tags(text_from_file_with_apath, topK=70, withWeight= True)
wordlist_after_jieba = jieba.analyse.extract_tags(text_from_file_with_apath, topK=70, withWeight= False)
print(wordlist_after_jieba)
label = ' '.join(wordlist_after_jieba)
# generate  wordcloud image
im_wordcloud = WordCloud(background_color = 'gray', font_path=font, width = 2000, height = 2000, max_words = 70).generate(label)
print(im_wordcloud)

plt.imshow(im_wordcloud)
plt.axis("off")
plt.show()


'''Data Precessing'''
# check missing values
# load the dataset
review_client = pd.read_csv('review_client.csv', encoding='gb2312')
print('Are there any missing values in review_client:\n', review_client.isna().any())
# drop rows with null review_client
review_client=review_client.dropna(axis=0)
# after remove rows with missing review_client
print('After remove rows with missing review_client:\n')
review_client.info()
# use heatmap to visualise above result
sns.heatmap(review_client.isnull(), cmap='viridis')
plt.show()
# check for duplications
print('Are there duplications in review_client:', review_client.duplicated().any())
# remove duplications
review_client.drop_duplicates(subset=['Review'], inplace=True)
# check for duplications again
print('After remove duplications:\n')
review_client.info()
# find ratings count outliers
sns.boxplot(x=review_client['Label'])
plt.show()
# transform categorical features into numerical features
le = preprocessing.LabelEncoder()
review_client['Client'] = le.fit_transform(review_client['Client'])
print(review_client['Client'][:5])
'''Text pre-processing'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
# use text_to_word_sequence to remove symbols
review_client['Review']= review_client['Review'].apply(text_to_word_sequence,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
MAX_WORDS = 6000
token = Tokenizer(num_words=MAX_WORDS)
# assigns unique int to all the words
token.fit_on_texts(review_client['Review'])
# generate a dictionary of words to numbers
word_index=token.word_index
print('Generated word_index:\n',word_index)
# check the original review_client
print('The original review_client:\n',review_client['Review'][0:5])
# convert corresponding words into index
review_client['Review'] = token.texts_to_sequences(review_client['Review'])
# check the processed review_client
print('After processing:\n',review_client['Review'][0:5])
# divide the data into attributes and labels
X = review_client['Review']
y = review_client['Label']
'''Text pre-processing'''
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.utils import np_utils
# split 80% of the data to the training set and 20% of the data to test set
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)
# the max and min length of reviews_client
review_length = [len(x) for x in X_train]
print('the max length of reviews_client：', max(review_length))
print('the min length of reviews_client：', min(review_length))
# take the first 500 words of each review to predict the rating and pad with 0 if length is less than 500
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)
# One-hot encoding of discrete features
y_train = np_utils.to_categorical(y_train-1, num_classes=10)
y_test = np_utils.to_categorical(y_test-1, num_classes=10)
# after one hot encoding
print('After one hot encoding:\n', y_train[0:5])
# Split the test set into test and validation set
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)

'''=====Machine Learning Model====='''
'''LinearRegression'''
lr = LinearRegression()
lr.fit(X_train, y_train)
# use the test data to check accuracy
predict_rating = lr.predict(X_test)
# evaluate the performance of the algorithm
print('MAE:', metrics.mean_absolute_error(y_test, predict_rating))
print('MSE:', metrics.mean_squared_error(y_test, predict_rating))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict_rating)))
print('R2_Score:',r2_score(y_test, predict_rating))

'''MLP'''
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.callbacks import ModelCheckpoint
# compile model
model = Sequential([
        Embedding(input_dim=6000, output_dim=16,input_length=500),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(5, activation='softmax')])
model.compile(optimizer='adam', metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()
# fit the model
model_checkpoint = ModelCheckpoint('best.hdf5', save_best_only=True)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
          batch_size=64, epochs=10, callbacks=[model_checkpoint])
# check the accuracy
model.load_weights('best.hdf5')
accracy = model.evaluate(X_test, y_test)
print('The accuracy of the model:', accracy[1])

