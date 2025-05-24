# %%
import numpy as np
import pandas as pd
import neattext.functions as nfx 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
df = pd.read_csv('udemy_course_data.csv')
df.head()

# %%
df['clean_title'] = df['course_title'].apply(nfx.remove_stopwords)
df['clean_title'] = df['clean_title'].apply(nfx.remove_special_characters)

df.head()

# %%
"""
Vectorise the Clean Title
"""

# %%
countvect = CountVectorizer()
cvmat = countvect.fit_transform(df['clean_title'])
cvmat

# %%
"""
Cosine Similary
"""

# %%
cossim = cosine_similarity(cvmat)
cossim

# %%
cossim.shape

# %%
"""
Recommend Course
"""

# %%
course_index = pd.Series(df.index, index=df['course_title']).drop_duplicates()
course_index

# %%
test = df[df['course_title'].str.contains('Profit')]
test.head()

# %%
top6 = test.sort_values(by='num_subscribers', ascending=False).head(6)
top6

# %%
index = course_index['Excel functions to analyze and visualize data']

scores = list(enumerate(cossim[index]))
scores

# %%
sorted_score = sorted(scores,key = lambda x:x[1],reverse=True)
sorted_score

# %%
sorted_indices = [i[0] for i in sorted_score[1:]]
sorted_values = [i[1] for i in sorted_score[1:]]

sorted_values

# %%
recommended_df = df.iloc[sorted_indices]
recommended_df

# %%
recommended_df['similarity_score'] = np.array(sorted_values)
recommended_df

# %%
usedf = recommended_df[['clean_title','similarity_score']]
usedf

# %%
def recommend_course(title,numrec = 10):
    
    course_index = pd.Series(
        df.index, index=df['course_title']).drop_duplicates()

    index = course_index[title]

    scores = list(enumerate(cossim [index]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    selected_course_index = [i[0] for i in sorted_scores[1:]]

    selected_course_score = [i[1] for i in sorted_scores[1:]]

    recdf = df.iloc[selected_course_index]

    recdf['similarity_score'] = selected_course_score

    recommends = recdf[[
        'course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]

    return recommends.head(numrec)

rec = recommend_course('Financial Statements Made Easy',20)

rec

# %%
"""
Save the cleaned file into a new file
"""

# %%
df.to_csv('udemy_cleaned.csv',index = None)