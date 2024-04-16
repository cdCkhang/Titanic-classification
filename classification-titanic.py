import xulydulieu as PreProcessDT
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = PreProcessDT.df
PreProcessDT.mapping_value(data)

# Lay tat ca cac dong va cot tru cot cuoi (Features )
X = data.iloc[:, :-1]
# X = data['Pclass','Mature','Gender']

# Lay tat ca dong cua cot cuoi ( Label )
y = data.iloc[:, -1]
# y = data['Survied]

# Chia du lieu thanh 2 set : training set va test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=50)

# KNN K-Nearest Neigbors Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knnaccuracy = knn.score(X_test,y_test)

# Naive Bayes Models
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnbaccuracy = gnb.score(X_test,y_test)

# Decision Tree Models
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dtaccuracy = dt.score(X_test,y_test)
# Accuracy
print("Accuracy [KNN-Model]:", knnaccuracy)
print("Accuracy [NB-Model]:", gnbaccuracy)
print("Accuracy [DecisionTree-Model]:", dtaccuracy)


# Truc quan hoa du lieu tinh chinh xac cua 3 Models
Models = ['KNN', 'Naive Bayes', 'Decision Tree']
Scores = [knnaccuracy, gnbaccuracy, dtaccuracy]
plt.bar(Models, Scores, color=['blue', 'green', 'orange'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Performance')
plt.ylim(0, 1)
plt.show()


# Truc quan hoa voi scatter plot
X = data[['Pclass', 'Mature', 'Gender']]
y = data['Survived']

survived = X[y == 1]
not_survived = X[y == 0]

plt.scatter(survived['Pclass'], survived['Mature'], color='blue', label='Survived')
plt.scatter(not_survived['Pclass'], not_survived['Mature'], color='red', label='Not Survived')

plt.xlabel('Pclass')
plt.ylabel('Mature')
plt.title('Scatter Plot of Pclass vs Mature')

plt.grid(True)
plt.show()
