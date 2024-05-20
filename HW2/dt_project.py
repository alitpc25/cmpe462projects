# Decision tree
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# read wdbc.data file
df = pd.read_csv('wdbc.data', header=None)

# Number of instances: 569
# Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

# Class distribution: 357 benign, 212 malignant

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Apply 5-fold cross validation
# Divide the dataset into 5 equal parts
# 1 part for testing and 4 parts for training
# Repeat the process 5 times and average the accuracy

# Split the dataset into 5 equal parts
part1 = df.iloc[0:114, :]
part2 = df.iloc[114:228, :]
part3 = df.iloc[228:342, :]
part4 = df.iloc[342:456, :]
part5 = df.iloc[456:570, :]
parts = [part1, part2, part3, part4, part5]
train_accuracy = []
test_accuracy = []
feature_importances_arr = []
# Apply 5-fold cross validation

for i in range(5):
    test = parts[i].reset_index(drop=True)
    train = pd.concat([part for j, part in enumerate(parts) if j != i]).reset_index(drop=True)
    y_train = train.iloc[:,1]
    X_train = train.iloc[:,2:]

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.savefig('decision_tree'+str(i)+'.pdf')
    plt.close()

    # trainig accuracy
    outputs = list(clf.predict(X_train))
    score = 0

    for j in range(len(outputs)):
        if outputs[j] == y_train[j]:
            score+=1
    train_accuracy.append(score/len(y_train))

    y_test = test.iloc[:,1]
    X_test = test.iloc[:,2:]
    outputs = list(clf.predict(X_test))
    score = 0

    for j in range(len(outputs)):
        if outputs[j] == y_test[j]:
            score+=1
    test_accuracy.append(score/len(y_test))

    feature_importances = pd.DataFrame(clf.feature_importances_,
                                index = X_train.columns, 
                                columns=['importance']).sort_values('importance', 
                                                                   ascending=False)

    feature_importances_arr.append(feature_importances)

print('Average train accuracy: ', np.mean(train_accuracy))
print('Average test accuracy: ', np.mean(test_accuracy))


avg_feature_importances = pd.DataFrame(np.zeros((30,1)), index = X_train.columns, columns=['importance'])
for i in range(5):
    for j in range(len(feature_importances_arr[0])):
        avg_feature_importances.loc[feature_importances_arr[i].index[j]] += feature_importances_arr[i].iloc[j][0]

avg_feature_importances = avg_feature_importances/5
avg_feature_importances = avg_feature_importances.sort_values('importance', ascending=False)
print(avg_feature_importances)

top5_features = avg_feature_importances.index[0:5]
top10_features = avg_feature_importances.index[0:10]
top15_features = avg_feature_importances.index[0:15]
top20_features = avg_feature_importances.index[0:20]

top_features_arr = [top5_features, top10_features, top15_features, top20_features]

# apply perceptron on top features
for t in range(len(top_features_arr)):
    test_accuracy = []
    for i in range(5):
        test = parts[i].reset_index(drop=True)
        train = pd.concat([part for j, part in enumerate(parts) if j != i]).reset_index(drop=True)
        y_train = train.iloc[:,1]
        X_train = train[top_features_arr[t]]

        clf = Perceptron(tol=1e-3, random_state=0)
        clf = clf.fit(X_train, y_train)

        y_test = test.iloc[:,1]
        X_test = test[top_features_arr[t]]
        outputs = list(clf.predict(X_test))
        score = 0

        for j in range(len(outputs)):
            if outputs[j] == y_test[j]:
                score+=1
        test_accuracy.append(score/len(y_test))

    print("Perceptron accuracy on top "+str((t+1)*5)+" features:")
    print(np.mean(test_accuracy))


# Feature selection using decision tree gives advantage to perceptron since
# the features are selected based on their importance and perceptron 
# gives weights to the features based on their importance and uses them to classify



# Random Forest

test_accuracy_arr = []
train_accuracy_arr = []
feature_importances_arr = []
n_estimators = [10, 20, 50, 100, 200, 500]
# Apply 5-fold cross validation

for k in range(len(n_estimators)):
    test_accuracy = []
    train_accuracy = []
    for i in range(5):
        test = parts[i].reset_index(drop=True)
        train = pd.concat([part for j, part in enumerate(parts) if j != i]).reset_index(drop=True)
        y_train = train.iloc[:,1]
        X_train = train.iloc[:,2:]

        clf = RandomForestClassifier(max_depth=5, n_estimators=n_estimators[k])
        clf = clf.fit(X_train, y_train)

        # trainig accuracy
        outputs = list(clf.predict(X_train))
        score = 0

        for j in range(len(outputs)):
            if outputs[j] == y_train[j]:
                score+=1
        train_accuracy.append(score/len(y_train))

        y_test = test.iloc[:,1]
        X_test = test.iloc[:,2:]
        outputs = list(clf.predict(X_test))
        score = 0

        for j in range(len(outputs)):
            if outputs[j] == y_test[j]:
                score+=1
        test_accuracy.append(score/len(y_test))

        feature_importances = pd.DataFrame(clf.feature_importances_,
                                    index = X_train.columns, 
                                    columns=['importance']).sort_values('importance', 
                                                                    ascending=False)

        feature_importances_arr.append(feature_importances)

    print("Random forests test accuracy with "+str(n_estimators[k])+" estimators:")
    print(np.mean(test_accuracy))
    print("Random forests train accuracy with "+str(n_estimators[k])+" estimators:")
    print(np.mean(train_accuracy))

    test_accuracy_arr.append(np.mean(test_accuracy))
    train_accuracy_arr.append(np.mean(train_accuracy))
    
# Plot test and train accuracy
plt.plot(n_estimators, test_accuracy_arr, label='Test accuracy')
plt.plot(n_estimators, train_accuracy_arr, label='Train accuracy')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()