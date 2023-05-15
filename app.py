#standard library import
import pandas as pd
import numpy as np

#for collaborative filtering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import correlation 

#code for displaying recommended books
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

#reading data
books_df=pd.read_csv(r'C:\Users\prati\Desktop\Project\recommendation system\dataset\Books.csv',low_memory=False)
ratings_df=pd.read_csv(r'C:\Users\prati\Desktop\Project\recommendation system\dataset\Ratings.csv',low_memory=False)
users_df=pd.read_csv(r'C:\Users\prati\Desktop\Project\recommendation system\dataset\Users.csv',low_memory=False)

#Handling missing and null values in book dataframe
books_df['Publisher'].fillna('Unknown', inplace=True)

#Handling missing and null values in ratings dataframe
ratings_df['Book-Rating'].fillna(0, inplace=True)

#Handling missing and null values in users dataframe
users_df['Age'].fillna(users_df['Age'].median(), inplace=True)

# Merging all the data in single dataframe
merged_dataset= ratings_df.merge(books_df,on='ISBN')
merged_dataset=merged_dataset.merge(users_df,on='User-ID')

#data manipulation and cleaning
# fixing error in year of publication column
year_map = {'DK Publishing Inc': 2000,'Gallimard':2003}

# Replace the values in the 'City' column with their corresponding integer values
merged_dataset['Year-Of-Publication'] = merged_dataset['Year-Of-Publication'].replace(year_map)

# Replacing age values above 90 with the mean value of the age column

 # Calculating the mean value for all values which are less than 90
mean_age = merged_dataset['Age'][merged_dataset['Age'] < 90].mean() 


# Replacimg the values which are above 90 and below 10 with mean values
merged_dataset.loc[merged_dataset['Age'] < 10, 'Age'] = mean_age  
merged_dataset.loc[merged_dataset['Age'] > 90, 'Age'] = mean_age  

# Replacing year values with the mean value of the year column

#typecasting Year-Of-Publication column from string to numerical format 
merged_dataset['Year-Of-Publication'] = pd.to_numeric(merged_dataset['Year-Of-Publication'], errors='coerce')

# Calculating the most frequent value occuring in column
most_frequent_value = merged_dataset['Year-Of-Publication'].value_counts().idxmax()

#replacing all the values in year column which are below 1900 and above 2017 with most frequent value.
merged_dataset.loc[merged_dataset['Year-Of-Publication'] < 1900, 'Year-Of-Publication'] = most_frequent_value   # Replace the values
merged_dataset.loc[merged_dataset['Year-Of-Publication'] > 2017, 'Year-Of-Publication'] = most_frequent_value   # Replace the values

#Finding 50 most popular books

#counting number of ratings each books have got.
count_ratings=merged_dataset.groupby('Book-Title')['Book-Rating'].count().reset_index()

#renaming column
count_ratings.rename(columns={'Book-Rating':'count_ratings'},inplace=True)

#finding mean rating for each books.
mean_rating = merged_dataset.loc[merged_dataset['Book-Rating'] > 0].groupby('Book-Title')['Book-Rating'].mean().reset_index()

#renaming column
mean_rating.rename(columns={'Book-Rating':'mean_ratings'},inplace=True)

#merging count_rating,mean_rating column 
book_mean_rating=pd.merge(count_ratings,mean_rating, on='Book-Title', how='inner')

#filtering books which have got atleast 150 ratings.
popular_books=book_mean_rating[book_mean_rating['count_ratings']>=150].sort_values('mean_ratings',ascending=False)

#Merging this popular_book dataframe with book_df
popular_books = popular_books.merge(merged_dataset,on='Book-Title').drop_duplicates('Book-Title')

#selecting required feature which will help us to display books on streamlit
popular_books=popular_books[['User-ID','ISBN','Book-Title','Book-Author','Year-Of-Publication','Image-URL-M','count_ratings','mean_ratings','Publisher']]


#Merging this popular_book dataframe with book_df
#popular_books = popular_books.merge(merged_dataset,on='Book-Title').drop_duplicates('Book-Title')

#selecting required feature which will help us to display books on streamlit
popular_books=popular_books[['User-ID','ISBN','Book-Title','Book-Author','Year-Of-Publication','Image-URL-M','count_ratings','mean_ratings','Publisher']]



# Set user-agent headers to simulate a web browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
}

# Set the number of columns in the grid
num_columns = 5

# Create a container for the grid layout
container = st.container()

# Iterate through each row in the dataframe
for index, row in popular_books.iterrows():
    # Get the values from the current row
    book_title = row['Book-Title']
    publisher = row['Publisher']
    image_url = row['Image-URL-M']

    # Download the image from the provided URL
    response = requests.get(image_url, headers=headers)

    # Convert the image content to PIL Image object
    image = Image.open(BytesIO(response.content))

    # Display the book image, title, and publisher in a column
    columns = container.columns(num_columns)
    book_image_col = columns[0]
    book_title_col = columns[1]
    publisher_col = columns[2]

    book_image_col.image(image, caption="Book Image")
    book_title_col.write("Book Title:")
    book_title_col.write(book_title)
    publisher_col.write("Publisher:")
    publisher_col.write(publisher)
