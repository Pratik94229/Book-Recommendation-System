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


# Page 1: Popular Books
def popular_books_page():
    st.title("Most Popular Books")
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

# Page 2: Item-Based
def item_based_page():
    #Filtering users who have given atleast 200 ratings
    xp_readers = merged_dataset.groupby('User-ID').count()['Book-Rating'] > 200

    #Index of all the users who have given atleast 200 ratings to books
    xp_readers_index = xp_readers[xp_readers].index

    #Filtering all these users in dataframe
    filtered_xp_users = merged_dataset[merged_dataset['User-ID'].isin(xp_readers_index)]


    #Filtering all the books which have got atleast 50 ratings
    rated_books = filtered_xp_users.groupby('Book-Title').count()['Book-Rating']>=50

    #Index of all the books who have got atleast 50 ratings 
    popular_books_index= rated_books[rated_books].index

    #Filtering all books in dataframe
    final_filtered_df = filtered_xp_users[filtered_xp_users['Book-Title'].isin(popular_books_index)]

    #Creating  Pivot Table
    book_pivot = final_filtered_df.pivot_table(columns='User-ID', index='Book-Title', values="Book-Rating")
    book_pivot.fillna(0, inplace=True)




    # List of book genres
    book_list = book_pivot.index

    # Implementing nearest neighbors algorithm which uses clustering based on euclidian distance.
    def Item_based_recomm(book_pivot=book_pivot,book_name=''):
       try:
          book_sparse = csr_matrix(book_pivot)

          #Implementing nearest neighbour algorithm
          model = NearestNeighbors(algorithm='auto')
          model.fit(book_sparse)
  
          #finding index of book
          for i in range(len(book_pivot.index)):
             if(book_pivot.index[i]==book_name):
                 book_index=i

          #prediction using knn
          distances, suggestions = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1))    
  
          #Top 5 prediction
          new_df = pd.DataFrame()
          for i in range(5):
              df = pd.DataFrame()
              df=merged_dataset[merged_dataset['Book-Title']==book_pivot.index[suggestions[0][i]]].drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-M']]
              new_df=new_df.append(df, ignore_index=True)
          return new_df  
       except:
           print("Enter correct or complete book name")
           return None


    # Single select
    selected_book = st.selectbox("Select book you like", book_list)
    st.write("You may also like:")

    # Set user-agent headers to simulate a web browser request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }

    # Set the number of columns in the grid
    num_columns = 5

    # Create a container for the grid layout
    container = st.container()
    
    books=Item_based_recomm(book_pivot,selected_book)
    # Iterate through each row in the dataframe
    for index, row in books.iterrows():
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




# Page 3: User-Based
def user_based_page():
    st.title("User-Based Recommendations")
    # Add your code for user-based recommendations here

# Page 4: Matrix Factorization
def matrix_factorization_page():
    st.title("Matrix Factorization")
    # Add your code for matrix factorization recommendations here

# Page 5: Hybrid Recommendation system
def hyb_recommendation_page():
    st.title("Hybrid Recommendation System")
    # Add your code for Hybrid Recommendation System recommendations here

# Create a dictionary of page names and corresponding function names
pages = {
    "Popularity Based Recommendation": popular_books_page,
    "Item-Item Based filtering": item_based_page,
    "User-User Based filtering": user_based_page,
    "Matrix Factorization": matrix_factorization_page,
    "Hybrid Recommendation System": matrix_factorization_page
}

# Sidebar
st.sidebar.title("Book Recommendation System")
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Run the selected page function
pages[selected_page]()
