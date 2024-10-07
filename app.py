from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__,static_folder='static')

# Load your dataset (replace the file path with your actual file path)
df = pd.read_csv(r"C:\Users\Chahat Chaudhary\OneDrive\Desktop\Amazon\csv\amazon.csv")

# Placeholder for data preprocessing
def preprocess_data(dataframe):
    dataframe['about_product'] = dataframe['about_product'].str.lower()
    dataframe['about_product'].fillna('', inplace=True)
    return dataframe

# Placeholder for the recommendation system
# Placeholder for the recommendation system
def recommend_products(dataframe, product_name, top_n=5):
    # Convert the product name to lowercase
    product_name = product_name.lower()
    
    # Check if the product_name exists in the dataframe
    if product_name not in dataframe['product_name'].str.lower().values:
        return pd.DataFrame()  # Return an empty DataFrame if the product is not found
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['about_product'])
    
    # Use linear_kernel instead of cosine_similarity for improved performance
    cosine_sim_product = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the product
    product_index = dataframe[dataframe['product_name'].str.lower() == product_name].index[0]
    
    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim_product[product_index]))
    
    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top N similar products
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Get the top N recommended products
    top_recommendations = dataframe.iloc[top_indices]
    return top_recommendations


# Preprocess the data
df = preprocess_data(df)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        # Convert the product name to lowercase
        product_name = request.form['product_name'].lower()
        recommendations = recommend_products(df, product_name)
        return render_template('recommendations.html', product_name=product_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
