import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load models and data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity_count = pickle.load(open('similarity_count.pkl', 'rb'))
similarity_tfidf = pickle.load(open('similarity_tfidf.pkl', 'rb'))

def recommend(movie, model_type='count'):
    if movie not in movies['title'].values:
        return ["Movie not found. Please try another title."]
    
    movie_index = movies[movies['title'] == movie].index[0]
    similarity = similarity_count if model_type == 'count' else similarity_tfidf
    distances = similarity[movie_index]
    top5 = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in top5:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_model = 'count'
    movie_name = ''
    
    if request.method == 'POST':
        movie_name = request.form.get('movie')
        selected_model = request.form.get('model')
        recommendations = recommend(movie_name, selected_model)

    return render_template('index.html',
                           movie_list=movies['title'].values,
                           recommendations=recommendations,
                           selected_model=selected_model,
                           movie_name=movie_name)

@app.route('/charts')
def charts():
    return render_template('charts.html')

if __name__ == '__main__':
    app.run(debug=True)
