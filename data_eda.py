#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:03:26 2020

@author: Tung
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk

from tabulate import tabulate
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv('salary_data_cleaned.csv')


def title_simplifier(title):
	if 'data scientist' in title.lower():
		return 'data scientist'
	elif 'data engineer' in title.lower():
		return 'data engineer'
	elif 'analyst' in title.lower():
		return 'analyst'
	elif 'machine learning' in title.lower():
		return 'mle'
	elif 'manager' in title.lower():
		return 'manager'
	elif 'director' in title.lower():
		return 'director'
	else:
		return np.nan

def seniority(title):
	if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
		return 'senior'
	elif 'jr' in title.lower() or 'jr.' in title.lower() or 'junior' in title.lower():
		return 'jr'
	else:
		return np.nan

# filling nan by 'NA'
df.fillna('NA', inplace=True)

# apply functions
df['job_simp'] = df['job_title'].apply(title_simplifier)

df['seniority'] = df['job_title'].apply(seniority)

# fix LA state
df['job_state'] = df['job_state'].apply(lambda x: x.strip() if x.strip().lower()!='los angeles' else 'CA')
# job length
df['desc_len'] = df['job_description'].apply(lambda x: len(x))

# competitor count
df['num_comp'] = df['competitors'].apply(lambda x: len(x.split(',')) if x!='-1' else 0)

# hourly wage to annual
df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly==1 else x.min_salary, axis=1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly==1 else x.max_salary, axis=1)

# fix company text
df['company_text'] = df['company_text'].apply(lambda x: x.replace('\n',''))

# stats and plots
df.columns

df.rating.hist()
plt.show()

df.avg_salary.hist()
plt.show()

df.age.hist()
plt.show()

df.desc_len.hist()
plt.show()

# boxplot
df.boxplot(column=['age','avg_salary','rating'])
plt.show()

df.boxplot(column='rating')
plt.show()

# correlation
df[['age','avg_salary','rating','desc_len']].corr()
plt.show()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df[['age','avg_salary','rating','desc_len','num_comp']].corr(),vmax=.3, center=0, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

cols = ['location', 'headquarters', 'size', 'type_of_ownership',
'industry', 'sector', 'revenue', 'company_text',
'job_state', 'same_state', 'age', 'python', 'r-studio', 'spark', 'aws',
'excel', 'scala', 'job_simp', 'seniority']

df_cat = df[cols]

for col in cols:
	cat_num = df_cat[col].value_counts(dropna=False)[:20]
	len_cat_num = len(cat_num)
	print('Graph for {}: top {}'.format(col, len_cat_num))
	ch = sns.barplot(x=cat_num.index, y=cat_num)
	ch.set_xticklabels(ch.get_xticklabels(), rotation=90)
	plt.show()

pivot1 = pd.pivot_table(df, index = 'job_simp', values = 'avg_salary')
print(tabulate(pivot1, headers='keys', tablefmt='psql'))

pivot2 = pd.pivot_table(df, index = ['job_simp','seniority'], values = 'avg_salary')
print(tabulate(pivot2, headers='keys', tablefmt='psql'))

pivot3 = pd.pivot_table(df, index = ['job_state','job_simp'], values = 'avg_salary').sort_values(by='avg_salary', ascending=False)
print(tabulate(pivot3, headers='keys', tablefmt='psql'))

pivot4 = pd.pivot_table(df, index = ['job_state','job_simp'], values = 'avg_salary', aggfunc='count').sort_values(by='avg_salary', ascending=False)
print(tabulate(pivot4, headers='keys', tablefmt='psql'))

pivot5 = pd.pivot_table(df[df['job_simp']=='data scientist'], index = ['job_state'], values = 'avg_salary').sort_values(by='avg_salary', ascending=False)
print(tabulate(pivot5, headers='keys', tablefmt='psql'))

#df_pivots = df[['rating','hourly','num_comp','employer_provided','industry','sector','revenue','type_of_ownership','python','r-studio','spark','aws','excel','scala']]


words = " ".join(df['job_description'])

def punctuation_stop(text):
    """remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered


words_filtered = punctuation_stop(words)

text = " ".join([ele for ele in words_filtered])

wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
wc.generate(text)

plt.figure(figsize=[10,10])
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()

