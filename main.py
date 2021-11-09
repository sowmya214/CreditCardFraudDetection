import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")

# PREPROCESSING DATA

data = pd.read_csv("creditcard.csv")
print(data.shape)
# print(type(data))

print("null and NaN values in dataset")
print("null: ", data.isnull().sum().sum())
print("nan: ", data.isna().sum().sum())

print("column names")
print(data.columns)

print("original dataset")
print("percentage of frauds: ", (len(data[data['Class'] == 1])/len(data)) * 100)
print("percentage of NO frauds: ", (len(data[data['Class'] == 0])/len(data)) * 100)
print()

# scaling data
rob_scaler = RobustScaler()
scaled_amount = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
scaled_time = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time','Amount'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)
# print(data)

# split data into train, test datasets
X = data.drop('Class', axis=1)
Y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=29, shuffle=True)
print("train and test dataset")
print("percentage of frauds in train: ", (len(y_train[y_train == 1])/len(y_train)) * 100)
print("percentage of frauds in test: ", (len(y_test[y_test == 1])/len(y_test)) * 100)
print()

# undersampling to balance training dataset
undersampler = RandomUnderSampler(sampling_strategy='majority')
x_train_sampled, y_train_sampled = undersampler.fit_resample(x_train, y_train)
print("rebalanced training dataset")
print("percentage of no frauds in train: ", (len(y_train_sampled[y_train_sampled == 0])/len(y_train_sampled)) * 100)
print("percentage of frauds in train: ", (len(y_train_sampled[y_train_sampled == 1])/len(y_train_sampled)) * 100)
print()

figure = plt.subplot()
correlation_matrix = x_train_sampled.join(y_train_sampled).corr()
sns.heatmap(correlation_matrix, cmap='coolwarm_r', annot_kws={'size':20})
figure.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
#plt.show()

training_data = x_train_sampled.join(y_train_sampled)

# removing outliers
for column in training_data.columns:
    if column in ['Class']:
        pass
    else:
        tmp = training_data[column].values
        q1, q3 = np.percentile(tmp, 25), np.percentile(tmp, 75)
        iqr = q3 - q1
        cut_off = iqr * 1.75
        lower_boundary, upper_boundary = q1 - cut_off, q3 + cut_off
        outliers = [x for x in tmp if x < lower_boundary or x > upper_boundary]
        training_data = training_data.drop(training_data[(training_data[column] > upper_boundary)].index)
        training_data = training_data.drop(training_data[(training_data[column] < lower_boundary)].index)
        # print('Number of Instances after outliers removal: {}'.format(len(training_data)))

x_train_sampled = training_data.drop('Class', axis=1)
y_train_sampled = training_data['Class']

# TRAINING CLASSIFIERS

# KNN

knn = KNeighborsClassifier()
knn.fit(x_train_sampled, y_train_sampled)
training_score = cross_val_score(knn, x_train_sampled, y_train_sampled, cv=5)
print("Classifiers: ", knn.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

knn_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
new_knn = GridSearchCV(KNeighborsClassifier(), knn_params)
new_knn.fit(x_train_sampled, y_train_sampled)
opt_knn = new_knn.best_estimator_
knn_score = cross_val_score(opt_knn, x_train_sampled, y_train_sampled, cv=5)
print("Classifiers: ", opt_knn.__class__.__name__, "Has a training score of", round(knn_score.mean(), 2) * 100, "% accuracy score")
print()

# LOGISTIC REGRESSION

lr = LogisticRegression()
lr.fit(x_train_sampled, y_train_sampled)
training_score = cross_val_score(lr, x_train_sampled, y_train_sampled, cv=5)
print("Classifiers: ", lr.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

lr_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1]}
new_lr = GridSearchCV(LogisticRegression(), lr_params)
new_lr.fit(x_train_sampled, y_train_sampled)
opt_lr = new_lr.best_estimator_
lr_score = cross_val_score(opt_lr, x_train_sampled, y_train_sampled, cv=5)
print("Classifiers: ", opt_lr.__class__.__name__, "Has a training score of", round(lr_score.mean(), 2) * 100, "% accuracy score")
print()


# LINEAR DISCRIMINANT ANALYSIS

lda = LinearDiscriminantAnalysis()
lda.fit(x_train_sampled, y_train_sampled)
training_score = cross_val_score(lda, x_train_sampled, y_train_sampled, cv=5)
print("Classifiers: ", lda.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

lda_params = {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': np.arange(0, 1, 0.01)}
new_lda = GridSearchCV(LinearDiscriminantAnalysis(), lda_params)
new_lda.fit(x_train_sampled, y_train_sampled)
opt_lda = new_lda.best_estimator_
lda_score = cross_val_score(opt_lda, x_train_sampled, y_train_sampled, cv=5)
print("Classifiers: ", opt_lda.__class__.__name__, "Has a training score of", round(lda_score.mean(), 2) * 100, "% accuracy score")
print()



# TESTING SCORES

score = knn.score(x_test, y_test) * 100
print("KNN test score: ", score)
score = opt_knn.score(x_test, y_test) * 100
print("opt KNN test score: ", score, "\n")

score = lr.score(x_test, y_test) * 100
print("Logistic Regression test score: ", score)
score = opt_lr.score(x_test, y_test) * 100
print("opt Logistic Regression test score: ", score, "\n")


score = lda.score(x_test, y_test) * 100
print("Linear Discriminant Analysis test score: ", score)
score = opt_lda.score(x_test, y_test) * 100
print("opt Linear Discriminant Analysis test score: ", score, "\n")


classifiers = [opt_knn, opt_lr, opt_lda]
for clf in classifiers:
    metrics.plot_roc_curve(clf, x_test, y_test)
    #.figure_.suptitle("ROC curve comparison")

plt.show()
















