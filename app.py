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

@st.cache(allow_output_mutation=True)
def seed():
    return {'seed': random.randint(0, 100)}

random_state = seed()

if st.sidebar.button('Reset points', key='123'):
    random_state['seed'] = random.randint(0, 100)

st.write('Seed = ', random_state['seed'])

cluster_std = st.sidebar.slider('Dispersion', 0.2, 3.0, 0.2, 0.2)
x, _ = make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=cluster_std, shuffle=True, random_state=random_state['seed'])

n_clusters = st.sidebar.selectbox('Number of clusters', range(1, 10))

kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, max_iter=300, random_state=111)
y_kmeans = kmeans.fit_predict(x)

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(x[:, 0], x[:, 1], s=100, c=kmeans.labels_, cmap='Set1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, marker='*', color='k')

st.pyplot(fig)


