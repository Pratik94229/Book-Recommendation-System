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



# Page 1: Popular Books page
def popular_books_page():
    st.title("Most Popular Books")
    # Setting user-agent headers to simulate a web browser request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }

    # Setting the number of columns in the grid
    num_columns = 5

    # Creating a container for the grid layout
    container = st.container()

    # Iterate through each row in the dataframe
    for index, row in popular_books.iterrows():
        # Getting the values from the current row
        book_title = row['Book-Title']
        publisher = row['Publisher']
        image_url = row['Image-URL-M']

        # Downloading the image from the provided URL
        response = requests.get(image_url, headers=headers)

        # Converting the image content to PIL Image object
        image = Image.open(BytesIO(response.content))

        # Displaying the book image, title, and publisher in a column
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
              # Dropping a row based on column value
              column_name = 'Book-Title'
              value_to_drop = book_name
              new_df = new_df[new_df[column_name] != value_to_drop]
          return new_df  
       except:
           print("Enter correct or complete book name")
           return None

    try:
        # Single select
        selected_book = st.selectbox("Select book user like", book_list)
        st.write("User may also like:")

        # Setting user-agent headers to simulate a web browser request
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
        }

        # Setting the number of columns in the grid
        num_columns = 5

        # Creating a container for the grid layout
        container = st.container()
    
        books=Item_based_recomm(book_pivot,selected_book)
        # Iterating through each row in the dataframe
        for index, row in books.iterrows():
            # Getting the values from the current row
            book_title = row['Book-Title']
            publisher = row['Publisher']
            image_url = row['Image-URL-M']

            # Downloading the image from the provided URL
            response = requests.get(image_url, headers=headers)

            # Converting the image content to PIL Image object
            image = Image.open(BytesIO(response.content))

            # Displaying the book image, title, and publisher in a column
            columns = container.columns(num_columns)
            book_image_col = columns[0]
            book_title_col = columns[1]
            publisher_col = columns[2]

            book_image_col.image(image, caption="Book Image")
            book_title_col.write("Book Title:")
            book_title_col.write(book_title)
            publisher_col.write("Publisher:")
            publisher_col.write(publisher)
    except:
        print('User information not found')



# Page 3: User-Based collaborative filtering 
def user_based_page():

    # Add your code for user-based recommendations here

    # Function to find the top N favorite book of a user 
    def favoritebook(activeUser,N):
        # 1. subset the dataframe to have the rows corresponding to the active user
        # 2. sort by the rating in descending order
        # 3. pick the top N rows
        topbooks=pd.DataFrame.sort_values(merged_dataset[merged_dataset['User-ID']==activeUser],['Book-Rating'],ascending=[0])[:N]
        # return the title corresponding to the books in topbooks 
        if(topbooks.empty):
            return "Insufficient data"
        else:
            return list(topbooks['Book-Title'])
        
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

    #finding  the K Nearest neighbours of a user and using their ratings to predict ratings of the active user for books they haven't rated.

    #Creating Pivot table
    userItemRatingMatrix=pd.pivot_table(final_filtered_df, values='Book-Rating',index=['User-ID'],
                                    columns=['Book-Title'])

    
    st.title("Select User for prediction")

    # Single select
    selected_user = st.selectbox("Select User ID for prediction",final_filtered_df['User-ID'].unique())

    # function to find the similarity between 2 users using correlation

    

    def similarity(user1,user2):
         #normalizing user1 by the mean rating of user 1 for any book for removing biases
         user1=np.array(user1)-np.nanmean(user1)

         #normalizing user1 by the mean rating of user 2 for any book for removing biases
         user2=np.array(user2)-np.nanmean(user2)
         # Now to find the similarity between 2 users
         # We'll first subset each user to be represented only by the ratings for book the 2 users have in common 
         commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
         # Gives us book for which both users have non NaN ratings 
         if len(commonItemIds)==0:
              # If there are no book in common 
              return 0
         else:
             user1=np.array([user1[i] for i in commonItemIds])
             user2=np.array([user2[i] for i in commonItemIds])
         return correlation(user1,user2)

    

    

    
        

    # Using this similarity function we will find the nearest neighbours of the active user

    def nearestNeighbourRatings(activeUser,K):
        try:
            #This function will find the K Nearest neighbours of the active user, then 
            #use their ratings to predict the activeUsers ratings for other movies 
    
            # Creating an empty matrix whose row index is userIds, and the value will be 
            # similarity of that user to the active User for finding similarity with other users.
            similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,
                                  columns=['Similarity'])
    
            # Finding the similarity between user i and the active user and add it to the similarityMatrix
            # using similarity function.
            for i in userItemRatingMatrix.index:
                similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        
            # Sorting the similarity matrix in the descending order of similarity    
            similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                              ['Similarity'],ascending=False)
    
            # Finding K Nearest neighbours of the active user
            nearestNeighbours=similarityMatrix[:K]
     
    
            # Using the nearest neighbours ratings to predict the active user's rating for every books
    
            neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    
            # A placeholder for the predicted item ratings. 
            predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])

            #We will find predicted rating for active user using the above formula

            # for each item 
            for i in userItemRatingMatrix.columns:
                # start with the average rating of the user
                predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])

                # for each neighbour in the neighbour list
                for j in neighbourItemRatings.index:

                 #If the neighbour has rated that item Add the rating of the neighbour for that item
            #adjusted by the average rating of the neighbour weighted by the similarity of the neighbour 
            #to the active user
                 if (userItemRatingMatrix.loc[j,i]>0):
                     
                     predictedRating += (userItemRatingMatrix.loc[j,i]
                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                # adding the rating to the predicted Rating matrix
                predictItemRating.loc[i,'Rating']=predictedRating
    
            return predictItemRating
        except:
            return None    
        
        # Using predicted Ratings to find the top N Recommendations for the active user 

    def topNRecommendations(activeUser,N):
      try:
        # Using the 10 nearest neighbours to find the predicted ratings
        predictItemRating=nearestNeighbourRatings(activeUser,10)
    
        #removing books which are already read by active user
        booksAlreadyRead=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    
        # finding the list of items whose ratings which are not NaN
        predictItemRating=predictItemRating.drop(booksAlreadyRead)
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
        # This will give us the list of itemIds which are the top recommendations 
        # Let's find the corresponding book titles 

        topRecommendationTitles=(final_filtered_df.loc[final_filtered_df['Book-Title'].isin(topRecommendations.index)])
        list(set(topRecommendationTitles['Book-Title']))
        return final_filtered_df[final_filtered_df['Book-Title'].isin(list(set(topRecommendationTitles['Book-Title'])))].drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-M']].reset_index().drop(['index'],axis=1)
      except:
        return None
      
    fav_books=pd.DataFrame([favoritebook(selected_user,10)]).T
    new_columns = {0:'Favourite Books'}
    fav_books = fav_books.rename(columns=new_columns)

    st.write(f'Most favourite books of users')
    st.write(fav_books)
    st.write("Recommended books for user based on liked books.")
    recommend=topNRecommendations(selected_user,10)

    # Setting user-agent headers to simulate a web browser request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }

    # Setting the number of columns in the grid
    num_columns = 5

    # Creating a container for the grid layout
    container = st.container()

    # Iterating through each row in the dataframe
    for index, row in recommend.iterrows():
        # Get the values from the current row
        book_title = row['Book-Title']
        publisher = row['Publisher']
        image_url = row['Image-URL-M']

        # Downloading the image from the provided URL
        response = requests.get(image_url, headers=headers)

        # Convert the image content to PIL Image object
        image = Image.open(BytesIO(response.content))

        # Displaying the book image, title, and publisher in a column
        columns = container.columns(num_columns)
        book_image_col = columns[0]
        book_title_col = columns[1]
        publisher_col = columns[2]

        book_image_col.image(image, caption="Book Image")
        book_title_col.write("Book Title:")
        book_title_col.write(book_title)
        publisher_col.write("Publisher:")
        publisher_col.write(publisher)
  


# Page 4: Matrix Factorization Page
def matrix_factorization_page():
    st.title("Latent Factor collaborative filtering")
    # Add your code for matrix factorization recommendations here

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

    #finding  the K Nearest neighbours of a user and using their ratings to predict ratings of the active user for books they haven't rated.

    #Creating Pivot table
    userItemRatingMatrix=pd.pivot_table(final_filtered_df, values='Book-Rating',index=['User-ID'],
                                    columns=['Book-Title'])

    # Function to predict ratings for all the users.User atleast 200 steps for for accurate prediction
    def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):
        # R is the user item rating matrix 
        # K is the number of factors we will find 
        # We'll be using Stochastic Gradient descent to find the factor vectors

        N=len(R.index)# Number of users
        M=len(R.columns) # Number of items 

        # This is the user factor matrix we want to find. It will has N rows on for each user and K columns,
        # one for each factor. We are initializing this matrix with some random numbers, then we will iteratively move towards 
        # the actual value we want to find 
        P=pd.DataFrame(np.random.rand(N,K),index=R.index)
        # This is the product factor matrix we want to find. It will have M rows, 
        # one for each product/item/movie.
        Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
        # This is the product factor matrix we want to find. It will have M rows, 
        # one for each product/item/movie. 

        # SGD will loop through the ratings in the user item rating matrix 
        # It will do this as many times as we specify (number of steps) or 
        # until the error we are minimizing reaches a certain threshold
        for step in range(steps):
            # SGD will loop through the ratings in the user item rating matrix 
            # It will do this as many times as we specify (number of steps) or 
            # until the error we are minimizing reaches a certain threshold 
            for i in R.index:
                for j in R.columns:
                    if R.loc[i,j]>0:
                        # For each rating that exists in the training set 
                        #Calulating the error for one rating (ie difference between the actual value of the rating 
                 #and the predicted value (dot product of the corresponding user factor vector and item-factor vector)
                 #which we have an error function to minimize

                        eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])

                    # The Ps and Qs should be moved in the downward direction 
                    # of the slope of the error at the current point 
                        P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])

                    # Gamma is the size of the step we are taking / moving the value of P by 
                    # The value in the brackets is the partial derivative of the error function ie the slope. 
                    # Lamda is the value of the regularization parameter which penalizes the model for the 
                    # number of factors we are finding.
                        Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])


        # checking the value of the error function to see if we have reached 
        # the threshold at which we want to stop, else we will repeat the process
            e=0
            for i in R.index:
                for j in R.columns:
                    if R.loc[i,j]>0:
                        #Sum of squares of the errors in the rating
                        e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
            if e<0.001:
                break
            #print(step)
        return P,Q
    
    # top 5 recommendations for a user 
    def matrix_fac_recommendation(user):
      try:
        (P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=10)

        #List of 20 active users
        #list(userItemRatingMatrix.index)[:20]
  
        #use these ratings to find top recommendations for a user
        activeUser=user
        predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Book-Rating'])
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Book-Rating'],ascending=[0])[:10]
        # We found the ratings of all movies by the active user and then sorted them to find the top 5 movies 
        topRecommendationTitles=final_filtered_df.loc[final_filtered_df['Book-Title'].isin(topRecommendations.index)]
        df=final_filtered_df[final_filtered_df['Book-Title'].isin(list(set(topRecommendationTitles['Book-Title'])))].drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-M']].reset_index().drop(['index'],axis=1)
        return df
      except:
        return None
  
    #Predicting for user 
    selected_user = st.selectbox("Select User ID for prediction",userItemRatingMatrix.index)
    recommend=matrix_fac_recommendation(selected_user)  

    # Setting user-agent headers to simulate a web browser request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }

    # Setting the number of columns in the grid
    num_columns = 5

    # Creating a container for the grid layout
    container = st.container()
    
    try:
        # Iterating through each row in the dataframe
        for index, row in recommend.iterrows():
            # Get the values from the current row
                book_title = row['Book-Title']
                publisher = row['Publisher']
                image_url = row['Image-URL-M']

                # Downloading the image from the provided URL
                response = requests.get(image_url, headers=headers)

                # Converting the image content to PIL Image object
                image = Image.open(BytesIO(response.content))

                # Displaying the book image, title, and publisher in a column
                columns = container.columns(num_columns)
                book_image_col = columns[0]
                book_title_col = columns[1]
                publisher_col = columns[2]

                book_image_col.image(image, caption="Book Image")
                book_title_col.write("Book Title:")
                book_title_col.write(book_title)
                publisher_col.write("Publisher:")
                publisher_col.write(publisher)
    except:
        print("Unable to predict due to insufficient information")            
  
 
# Page 5: Hybrid Recommendation system
def hyb_recommendation_page():
    st.title("Hybrid Recommendation System")
    # Add your code for Hybrid Recommendation System recommendations

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
              # Drop a row based on column value
              column_name = 'Book-Title'
              value_to_drop = book_name
              new_df = new_df[new_df[column_name] != value_to_drop]
          return new_df  
       except:
           print("Enter correct or complete book name")
           return None


     # Function to find the top N favorite book of a user 
    def favoritebook(activeUser,N):
        # 1. subset the dataframe to have the rows corresponding to the active user
        # 2. sort by the rating in descending order
        # 3. pick the top N rows
        topbooks=pd.DataFrame.sort_values(merged_dataset[merged_dataset['User-ID']==activeUser],['Book-Rating'],ascending=[0])[:N]
        # return the title corresponding to the books in topbooks 
        if(topbooks.empty):
            return "Insufficient data"
        else:
            return list(topbooks['Book-Title'])
        
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

    #finding  the K Nearest neighbours of a user and using their ratings to predict ratings of the active user for books they haven't rated.

    #Creating Pivot table
    userItemRatingMatrix=pd.pivot_table(final_filtered_df, values='Book-Rating',index=['User-ID'],
                                    columns=['Book-Title'])

    
 

    # function to find the similarity between 2 users using correlation

    

    def similarity(user1,user2):
         #normalizing user1 by the mean rating of user 1 for any book for removing biases
         user1=np.array(user1)-np.nanmean(user1)

         #normalizing user1 by the mean rating of user 2 for any book for removing biases
         user2=np.array(user2)-np.nanmean(user2)
         # Now to find the similarity between 2 users
         # We'll first subset each user to be represented only by the ratings for book the 2 users have in common 
         commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
         # Gives us book for which both users have non NaN ratings 
         if len(commonItemIds)==0:
              # If there are no book in common 
              return 0
         else:
             user1=np.array([user1[i] for i in commonItemIds])
             user2=np.array([user2[i] for i in commonItemIds])
         return correlation(user1,user2)

    

    

    
        

    # Using this similarity function we will find the nearest neighbours of the active user

    def nearestNeighbourRatings(activeUser,K):
        try:
            #This function will find the K Nearest neighbours of the active user, then 
            #use their ratings to predict the activeUsers ratings for other movies 
    
            # Creating an empty matrix whose row index is userIds, and the value will be 
            # similarity of that user to the active User for finding similarity with other users.
            similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,
                                  columns=['Similarity'])
    
            # Finding the similarity between user i and the active user and add it to the similarityMatrix
            # using similarity function.
            for i in userItemRatingMatrix.index:
                similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        
            # Sorting the similarity matrix in the descending order of similarity    
            similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                              ['Similarity'],ascending=False)
    
            # Finding K Nearest neighbours of the active user
            nearestNeighbours=similarityMatrix[:K]
     
    
            # Using the nearest neighbours ratings to predict the active user's rating for every books
    
            neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    
            # A placeholder for the predicted item ratings. 
            predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])

            #We will find predicted rating for active user using the above formula

            # for each item 
            for i in userItemRatingMatrix.columns:
                # start with the average rating of the user
                predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])

                # for each neighbour in the neighbour list
                for j in neighbourItemRatings.index:

                 #If the neighbour has rated that item Add the rating of the neighbour for that item
            #adjusted by the average rating of the neighbour weighted by the similarity of the neighbour 
            #to the active user
                 if (userItemRatingMatrix.loc[j,i]>0):
                     
                     predictedRating += (userItemRatingMatrix.loc[j,i]
                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                # adding the rating to the predicted Rating matrix
                predictItemRating.loc[i,'Rating']=predictedRating
    
            return predictItemRating
        except:
            return None    
        





    # Using predicted Ratings to find the top N Recommendations for the active user 

    def topNRecommendations(activeUser,N):
      try:
        # Using the 10 nearest neighbours to find the predicted ratings
        predictItemRating=nearestNeighbourRatings(activeUser,10)
    
        #removing books which are already read by active user
        booksAlreadyRead=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    
        # finding the list of items whose ratings which are not NaN
        predictItemRating=predictItemRating.drop(booksAlreadyRead)
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
        # This will give us the list of itemIds which are the top recommendations 
        # Let's find the corresponding book titles 

        topRecommendationTitles=(final_filtered_df.loc[final_filtered_df['Book-Title'].isin(topRecommendations.index)])
        list(set(topRecommendationTitles['Book-Title']))
        return final_filtered_df[final_filtered_df['Book-Title'].isin(list(set(topRecommendationTitles['Book-Title'])))].drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-M']].reset_index().drop(['index'],axis=1)
      except:
        return None    
    
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

    #finding  the K Nearest neighbours of a user and using their ratings to predict ratings of the active user for books they haven't rated.

    #Creating Pivot table
    userItemRatingMatrix=pd.pivot_table(final_filtered_df, values='Book-Rating',index=['User-ID'],
                                    columns=['Book-Title'])
    






    # Function to predict ratings for all the users.User atleast 200 steps for for accurate prediction
    def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):
        # R is the user item rating matrix 
        # K is the number of factors we will find 
        # We'll be using Stochastic Gradient descent to find the factor vectors

        N=len(R.index)# Number of users
        M=len(R.columns) # Number of items 

        # This is the user factor matrix we want to find. It will has N rows on for each user and K columns,
        # one for each factor. We are initializing this matrix with some random numbers, then we will iteratively move towards 
        # the actual value we want to find 
        P=pd.DataFrame(np.random.rand(N,K),index=R.index)
        # This is the product factor matrix we want to find. It will have M rows, 
        # one for each product/item/movie.
        Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
        # This is the product factor matrix we want to find. It will have M rows, 
        # one for each product/item/movie. 

        # SGD will loop through the ratings in the user item rating matrix 
        # It will do this as many times as we specify (number of steps) or 
        # until the error we are minimizing reaches a certain threshold
        for step in range(steps):
            # SGD will loop through the ratings in the user item rating matrix 
            # It will do this as many times as we specify (number of steps) or 
            # until the error we are minimizing reaches a certain threshold 
            for i in R.index:
                for j in R.columns:
                    if R.loc[i,j]>0:
                        # For each rating that exists in the training set 
                        #Calulating the error for one rating (ie difference between the actual value of the rating 
                 #and the predicted value (dot product of the corresponding user factor vector and item-factor vector)
                 #which we have an error function to minimize

                        eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])

                    # The Ps and Qs should be moved in the downward direction 
                    # of the slope of the error at the current point 
                        P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])

                    # Gamma is the size of the step we are taking / moving the value of P by 
                    # The value in the brackets is the partial derivative of the error function ie the slope. 
                    # Lamda is the value of the regularization parameter which penalizes the model for the 
                    # number of factors we are finding.
                        Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])


        # checking the value of the error function to see if we have reached 
        # the threshold at which we want to stop, else we will repeat the process
            e=0
            for i in R.index:
                for j in R.columns:
                    if R.loc[i,j]>0:
                        #Sum of squares of the errors in the rating
                        e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
            if e<0.001:
                break
            #print(step)
        return P,Q
    
    # top 5 recommendations for a user 
    def matrix_fac_recommendation(user):
      try:
        (P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=10)

        #List of 20 active users
        #list(userItemRatingMatrix.index)[:20]
  
        #use these ratings to find top recommendations for a user
        activeUser=user
        predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Book-Rating'])
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Book-Rating'],ascending=[0])[:10]
        # We found the ratings of all movies by the active user and then sorted them to find the top 5 movies 
        topRecommendationTitles=final_filtered_df.loc[final_filtered_df['Book-Title'].isin(topRecommendations.index)]
        df=final_filtered_df[final_filtered_df['Book-Title'].isin(list(set(topRecommendationTitles['Book-Title'])))].drop_duplicates('Book-Title')[['Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-M']].reset_index().drop(['index'],axis=1)
        return df
      except:
        return None
      
      
    





















#Implementing hybrid recommendation system

    def Hybrid_recommender(user_id,book_name,popular_books):
       #finding author of the book
       author=books_df[books_df['Book-Title']==book_name]['Book-Author'].unique()[0]

       #Radomly selecting 3 popular books
       result=popular_books[~(popular_books['Book-Title']==book_name)].sample(n=3)

       #popular books based on similar author
       filter_author_books=result[result['Book-Author']==author]
       filter_author_books=filter_author_books[~(filter_author_books['Book-Title']==book_name)].head(2)

       #concating two dataframe(20% weight to both)
       result = pd.concat([result,filter_author_books], ignore_index=True).drop_duplicates('Book-Title')
  

       #using item based collaborative filtering(20% weight)
       recom_df=Item_based_recomm(book_pivot,book_name)
       recom_df=recom_df[~(recom_df['Book-Title']==book_name)].head(2)
       if recom_df is None:
         pass
       else:  
         #concating two dataframe
         result = pd.concat([result,recom_df], ignore_index=True).drop_duplicates('Book-Title')

       #using user based filtering(30% weight)
       user_df=topNRecommendations(user_id,3)
       user_df=user_df[~(user_df['Book-Title']==book_name)]
       if user_df is None:
        pass
       else:
         #concating two dataframe
         result = pd.concat([result,user_df], ignore_index=True).drop_duplicates('Book-Title')
  
       #using matrix factorization(40% weight)
       try:
         matrix_df=matrix_fac_recommendation(user_id)
         matrix_df=matrix_df.head(4)
       except:
         return None
       if matrix_df is None:
         pass
       else:
         #concating two dataframe
          result = pd.concat([result, matrix_df], ignore_index=True).drop_duplicates('Book-Title')
  
       return result[['Book-Title','Book-Author','Year-Of-Publication','Image-URL-M','Publisher']].head(10)
    
    #Predicting for user 
    selected_user = st.selectbox("Select User ID for prediction",userItemRatingMatrix.index)
    selected_book = st.selectbox("Select book liked by user",userItemRatingMatrix.columns)
    recommend=Hybrid_recommender(selected_user,selected_book,popular_books)  

    # Setting user-agent headers to simulate a web browser request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
    }

    # Setting the number of columns in the grid
    num_columns = 5

    # Creating a container for the grid layout
    container = st.container()
    
    try:
        # Iterating through each row in the dataframe
        for index, row in recommend.iterrows():
            # Getting the values from the current row
                book_title = row['Book-Title']
                publisher = row['Publisher']
                image_url = row['Image-URL-M']

                # Downloading the image from the provided URL
                response = requests.get(image_url, headers=headers)

                # Converting the image content to PIL Image object
                image = Image.open(BytesIO(response.content))

                # Displaying the book image, title, and publisher in a column
                columns = container.columns(num_columns)
                book_image_col = columns[0]
                book_title_col = columns[1]
                publisher_col = columns[2]

                book_image_col.image(image, caption="Book Image")
                book_title_col.write("Book Title:")
                book_title_col.write(book_title)
                publisher_col.write("Publisher:")
                publisher_col.write(publisher)
    except:
        print("Unable to predict due to insufficient information")    


# Creating a dictionary of page names and corresponding function names
pages = {
    "Popularity Based Recommendation": popular_books_page,
    "Item Based collaborative filtering": item_based_page,
    "User Based collaborative filtering": user_based_page,
    "Matrix Factorization": matrix_factorization_page,
    "Hybrid Recommendation System": hyb_recommendation_page
}

# code for Sidebar
st.sidebar.title("Book Recommendation System")
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Running the selected page function
pages[selected_page]()

