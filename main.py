from flask import Flask
from collections import Counter
import math
import pandas as pd
from sklearn import preprocessing
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/recommentdations")
def getRecommendations():
    user_qu = [1,2,2,2,1,3,2,1,2,1] # feature vector for The Post
    recommended_cat = recommendedCategories(user_query=user_qu, k_recommendations=5)
    print(json.dumps(recommended_cat))
    return json.dumps(recommended_cat)

## Return recomentations
def recommendedCategories(user_query,k_recommendations):
    df = pd.read_csv(r"C:\Users\ACER\Desktop\dulitha_formated.csv")
    
    # seperate multi valued food type
    df['foodType'] = df['foodType'].str.split(',')
    
    df = (df
     .set_index(['age','nationality','isVegetarian','priceRange','isTakeAway','frequency','isIndividual','isLikeDriveThrough','time','paymentMethod'])['foodType']
     .apply(pd.Series)
     .stack()
     .reset_index()
     .rename(columns={0:'foodType'}))
    
    del df['level_10']
    #df.head()
    
    # Prepare the data for use in the knn algorithm by picking
    # the relevant columns and converting the numeric columns
    # to numbers since they were read in as strings
    cat_recommendation_data = []
    for row in df.values:
        data_row = list(map(float, row[2:]))
        cat_recommendation_data.append(data_row)
    
    ## use KNN algorythm to get most matching categories
    recommendation_indices, _ = knn(
        cat_recommendation_data, user_query, k=k_recommendations,
        distance_fn=euclidean_distance, choice_fn=lambda x: None
    )
    
    ## get category name from results
    cat_recommendations = []
    for row in recommendation_indices:
        cat_recommendations.append(row[1])

    return cat_recommendations

# Define knn algorythm
def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[:-1], query)
        
        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)


if __name__ == '__main__':
    app.run(debug=True)
