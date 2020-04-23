import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, classification_report
import tree
from sklearn import tree as sktree
from sklearn.utils import Bunch
import tree

DEPTH = 10
N_FEATURES = 5

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = DEPTH

    def fit(self, X, y):
        self.n_classes = 2
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):   #predict class
        return [self._predict(inputs) for inputs in X]

    def debug(self, feature_names, class_names, show_details=False):
        """Print ASCII visualization of decision tree."""
        self.tree.debug(feature_names, class_names, show_details)

    def _gini(self, y):  #compute gini impurity
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes))

    def _best_split(self, X, y): #find the best split, i.e. average weighted impurity is minimized as much as possible
        #METHOD:
        # 1. Iterate through the sorted feature values as possible thresholds
        # 2. Keep track of the number of samples per class on the left and on the right
        # 3. Increment/decrement them by 1 after each threshold.
        m = y.size
        if m <= 1: #need at least two elements to split
            return None, None

        #count up number in each class
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]

        #compute gini score of current node
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        #Iterate through all features
        for idx in range(self.n_features):
            # Sort data along selected feature
            thresholds,classes = X[:, idx], y[:, 0]

            #compute iteratively
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]: #don't split at identical values
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        count = 0
        while count <= N_FEATURES:
            count = count + 1
            #build decision tree by recursively finding the best split
            num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
            predicted_class = np.argmax(num_samples_per_class)
            node = tree.Node(
                gini=self._gini(y),
                num_samples=y.size,
                num_samples_per_class=num_samples_per_class,
                predicted_class=predicted_class,
            )

            if depth < self.max_depth:
                idx, thr = self._best_split(X, y)
                if idx is not None:
                    indices_left = X[:, idx] < thr
                    X_left, y_left = X[indices_left], y[indices_left]
                    X_right, y_right = X[~indices_left], y[~indices_left]
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                    node.right = self._grow_tree(X_right, y_right, depth + 1)
            #print(node.feature_index )
            return node

    def _predict(self, inputs):
        #predict class for single entry
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


def test_accuracy(pred_score, predlabels):
    print(N_FEATURES, 'features')
    y_true = predlabels
    cm = confusion_matrix([1-x for x in y_true], [1-int(x > 0.5) for x in pred_score])
    print('Sensitivity: {}'.format(float(cm[0][0])/(cm[0][0]+cm[0][1])))
    print('Specificity: {}'.format(float(cm[1][1])/(cm[1][1]+cm[1][0])))
    recall = ((cm[0][0])/(cm[0][0]+cm[0][1]) + (cm[1][1])/(cm[1][1]+cm[1][0]))/2
    print('Recall: {}'.format(float(recall)))
    #print ('class report')
    #print (classification_report([1-x for x in y_true], [1-int(x > 0.5) for x in pred_score]))
    print('Confusion matrix:')
    print(cm)

if __name__ == "__main__":
    DEPTH = 10
    N_FEATURES = 5

    attributes_train = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7',
                        'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14',
                        'question15', 'question16', 'question17', 'question18', 'question19', 'question20',
                        'question21', 'question22', 'question23', 'question24', 'question25', 'question26',
                        'question27', 'question28', 'question29', 'question30', 'diag']
    X_train_labels = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7',
                        'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14',
                        'question15', 'question16', 'question17', 'question18', 'question19', 'question20',
                        'question21', 'question22', 'question23', 'question24', 'question25', 'question26',
                        'question27', 'question28', 'question29', 'question30']
    y_train_labels = ['diag']

    attributes_test = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7',
                       'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14',
                       'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21',
                       'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28',
                       'question29', 'question30', 'ASD']
    X_test_labels = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7',
                       'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14',
                       'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21',
                       'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28',
                       'question29', 'question30']
    y_test_labels = ['ASD']

    train, test = pd.read_csv('primary_dataset.csv', usecols=attributes_train), pd.read_csv('validation_dataset.csv',usecols=attributes_test)
    hide_details = False

    # Load data and store it into pandas DataFrame objects
    train_set = pd.DataFrame(train)
    #print(train_set)
    train_set = Bunch(
        data=train.loc[:, X_train_labels],
        target=train.loc[:, y_train_labels],
        feature_names = ["question {}".format(i) for i in range(1, 31)],
        target_names = ["diag {}".format(i) for i in range(0, 2)],
        hide_details=False
    )
    X_train, y_train = train_set.data.to_numpy(), train_set.target.to_numpy()
    for i in range(len(y_train)):
        if y_train[i]=='asd':
            y_train[i] = 1
        else:
            y_train[i] = 0
    # X_train = X_train.loc[:, X_train.columns != 'asd']
    # y_train = pd.DataFrame(X_train.loc(X_train.columns == 'asd'))

    X_test = pd.DataFrame(test)
    X_test = Bunch(
        data=test.loc[:, X_test_labels],
        target=test.loc[:, y_test_labels]
    )
    X_test, y_test = X_test.data, X_test.target



    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    #test_accuracy(list(clf.fit(X_train, y_train)), y_train)

    clf.debug(
        list(train_set.feature_names),
        list(train_set.target_names),
        not train_set.hide_details,
    )

    #sktree.plot_tree(clf,
    #                     feature_names=X_train_labels,
    #                     class_names=y_train_labels,
    #                     filled=True)
    #fig.savefig('tree.png')

    #clf.predict(X_test)
    #print(train.shape, test.shape)


    #clf = DecisionTreeClassifier(max_depth=DEPTH)
    #clf.fit(X_train, y_train)
