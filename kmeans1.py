import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import random

st.write("""
# Clustering using k-means
by Ariel Cedola
""")

if st.sidebar.button('Reset points'):
    random_state = random.randint(0, 100)
else:
    random_state = 100
st.sidebar.write('Seed = ', random_state)

cluster_std = st.sidebar.slider('Dispersion', 0.2, 3.0, 0.2, 0.2)
x, _ = make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=cluster_std, shuffle=True, random_state=random_state)

n_clusters = st.sidebar.selectbox('Number of clusters', range(1, 10))

#fig1, ax = plt.subplots(figsize=(12,8))
#plt.scatter(x[:, 0], x[:, 1], s=100)

#st.pyplot(fig1)

kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, max_iter=300, random_state=111)
y_kmeans = kmeans.fit_predict(x)

fig2, ax = plt.subplots(figsize=(12,8))
plt.scatter(x[:, 0], x[:, 1], s=100, c=kmeans.labels_, cmap='Set1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, marker='*', color='k')

st.pyplot(fig2)

#a = 100
#if st.button('Reset points', key='123'):
#    a = random.randint(0, 100)
#else:
#    a = 100

@st.cache
def seedxx():
    return random.randint(0, 100)# + a
#st.write('Seed = ', seedxx())

@st.cache(allow_output_mutation=True)
def seed2():
    return {'seed3': random.randint(0, 100)}

seed = seed2()

if st.button('Reset points', key='123'):
    seed['seed3'] = random.randint(0, 100)

st.write('Seed = ', seed['seed3'])

"""
if st.button('Reset points', key='123'):
    random_state = random.randint(0, 100)
else:
    random_state = 100
st.write('Seed = ', random_state)

cluster_std = st.slider('Dispersion', 0.2, 3.0, 0.2, 0.2, key='asd')
x, _ = make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=cluster_std, shuffle=True, random_state=random_state)

n_clusters = st.selectbox('Number of clusters', range(1, 10), key='zxc')

fig1, ax = plt.subplots(figsize=(12,8))
plt.scatter(x[:, 0], x[:, 1], s=100)

st.pyplot(fig1)
"""