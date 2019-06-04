from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# note that I am assuming the target is in the last position of the dataframe
# additionally, I am assuming that the list has already been filtered(ie. we are only training on the top players)
# additionally, my current assumption is the data has already been transformed into non-categorical data
def train_model(df, max_k):
    X = df.iloc[:, :-1].copy()#assuming our target is at the very end of the dataframe
    y = df.iloc[:, -1].copy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= .2)
    k_scores = []
    for k in range(1,max_k):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(cv_scores.mean())

    max_score = max(k_scores)

    #     get the arg max of the score array to determine the optimnal value for k
    k_opt = None
    for i in range(len(k_scores)):
        if k_scores[i] == max_score:
            k_opt = i + 1

    knn = KNeighborsClassifier(n_neighbors=k_opt)
    kf = KFold(n_splits=10)

    # split the training data into different validation folds and run a training loop on it

    kf.get_n_splits(x_train)
    training_scores = []


    knn.fit(x_train, y_train)#we get about -2.something places per map accuracy
    # testing_accuracy = (knn.predict(x_test) - y_test).mean()

    predictions = knn.predict(x_test)
    correct_predictions = predictions[predictions == y_test]

    print("TESTING ACCURACY: ", len(correct_predictions) / len(predictions))
    return knn


def player_path_model(df, max_k):
    #TODO: get x and y from df
    
    # Zone Data
    x = None

    # Player Position Category
    y = None
    
    best_score = 0
    best_model = None
    
    #Hyperparameter Tuning
    for k in range(max_k):
        vec = DictVectorizer(sparse=False)
        scaler = StandardScaler()
        model = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([('vec', vec), 
                            ('scaler', scaler),
                            ('fit', model)])
        
        score = cross_val_score(pipeline, 
                                x, 
                                y,
                                cv=5, scoring='accuracy').mean()

        if score > best_score or best_model is None:
            best_score = score
            best_model = pipeline

    print("Best Accuracy Score: " + str(best_score))
    return best_model
        
# preprocess the dataframe
def preprocess_data(drop_data):
    drop_data = drop_data.dropna()
    drop_data = drop_data.drop(columns = ['player'])#probably don't need to include the player in the model
    drop_data = drop_data.drop(columns = ['drop_loc_raw'])#probably don't need to include the player in the model
    drop_data = drop_data.dropna()
    drop_data = drop_data[drop_data['rank'] <= 5]
    labelencoder_X=LabelEncoder()
    X = drop_data.iloc[:,:].values
    drop_data['flight_path'] = labelencoder_X.fit_transform(X[:, 1])
    drop_data['map'] = labelencoder_X.fit_transform(X[:, 2])
    drop_data = drop_data.drop(columns = ['rank'])
    drop_data = drop_data[['flight_path', 'map', 'drop_loc_cat']]
    scaler = MinMaxScaler()
    drop_data.loc[:,:-1] = scaler.fit_transform(drop_data[drop_data.columns[:-1]])
    return drop_data


