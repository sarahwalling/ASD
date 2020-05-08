from random import seed
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def convert_class_to_int(dataset, column):
    for row in dataset:
        if row[column] == 'asd':
            row[column] = 1
        else:
            row[column] = 0

# split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        split.append(fold)
    return split


# calculate accuracy metrics
def get_accuracy(actual, predicted):
    confusion_matrix = a = [[0 for x in range(2)] for y in range(2)]
    correct = 0
    scores = list()
    for i in range(len(actual)):
        if actual[i] == 0:
            if predicted[i] == 0:
                confusion_matrix[0][0] = confusion_matrix[0][0] + 1
            if predicted[i] == 1:
                confusion_matrix[0][1] = confusion_matrix[0][1] + 1
        if actual[i] == 1:
            if predicted[i] == 0:
                confusion_matrix[1][0] = confusion_matrix[1][0] + 1
            if predicted[i] == 1:
                confusion_matrix[1][1] = confusion_matrix[1][1] + 1
        if actual[i] == predicted[i]:
            correct += 1
    sensitivity = float(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])) * 100
    specificity = float(confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])) * 100
    accuracy = correct / float(len(actual)) * 100.0
    scores.append(sensitivity)
    scores.append(specificity)
    scores.append(accuracy)
    return scores


# evaluate using the k-folds cross validation split
def evaluate_folds(dataset, n_folds, max_depth, min_size, sample_size, n_trees, n_features):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = random_forest(train_set, test_set, max_depth, min_size, sample_size, n_trees, n_features)
        actual = [row[-1] for row in fold]
        accuracy = get_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores

# evaluate with separate training and test datasets instead of k-folds
# this currently requires test set and training set to be same size
def evaluate_separate(dataset, test_set, max_depth, min_size, sample_size, n_trees, n_features):
    scores = list()
    predicted = random_forest(dataset, test_set, max_depth, min_size, sample_size, n_trees, n_features)
    actual = [row[-1] for row in dataset]
    accuracy = get_accuracy(actual, predicted)
    scores.append(accuracy)
    return scores

# split the dataset based on attribute and attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# calculate the gini impurity
def gini_index(groups, classes):
    #count all samples at the split point
    n_instances = float(sum([len(group) for group in groups]))

    #sum the weighted gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # don't divide by 0
        if size == 0:
            continue
        score = 0.0
        # calculate the score for the group from the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the score by group size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def to_leaf(group):
    leaves = [row[-1] for row in group]
    return max(set(leaves), key=leaves.count)


# create children splits for a node, or make leaf
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    #check for no split
    if not left or not right:
        node['left'] = node['right'] = to_leaf(left + right)
        return

    #check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_leaf(left), to_leaf(right)
        return

    #do left child
    if len(left) <= min_size:
        node['left'] = to_leaf(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)

    #do right child
    if len(right) <= min_size:
        node['right'] = to_leaf(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# find the best split point
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    new_index, new_value, new_score, new_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < new_score:
                new_index, new_value, new_score, new_groups = index, row[index], gini, groups
    return {'index': new_index, 'value': new_value, 'groups': new_groups}


# build a single decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# make a prediction with a single decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# create a random sample (with replacement)
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# random forest - calls helper methods to actually build it
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


seed(5)
attributes_train = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7',
                    'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14',
                    'question15', 'question16', 'question17', 'question18', 'question19', 'question20',
                    'question21', 'question22', 'question23', 'question24', 'question25', 'question26',
                    'question27', 'question28', 'question29', 'question30', 'diag']

attributes_test = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7',
                   'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14',
                   'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21',
                   'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28',
                   'question29', 'question30', 'ASD']

train, test = pd.read_csv('primary_dataset.csv', usecols=attributes_train), pd.read_csv('validation_dataset.csv', usecols=attributes_test)
dataset = train.values.tolist()
test_set = test.values.tolist()

# convert class column to integers
convert_class_to_int(dataset, len(dataset[0]) - 1)

# evaluate algorithm
n_folds = 3
max_depth = 10
min_size = 1
sample_size = 1.0
n_trees = 5
mean_sensitivity = list()
mean_specificity = list()
mean_accuracy = list()
num_features = [3,4,5,6,7,8,9,10]

print('Trees: %d\n' % n_trees)
for n_features in num_features:
    scores = evaluate_folds(dataset, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    #scores = evaluate_separate(dataset, test_set, max_depth, min_size, sample_size, n_trees, n_features)
    sensitivity = list()
    specificity = list()
    accuracy = list()
    print('Features: %d' % n_features)
    for row in scores:
        sensitivity.append(row[0])
        specificity.append(row[1])
        accuracy.append(row[2])
    print('Mean Sensitivity: %.3f%%' % (sum(sensitivity) / float(len(sensitivity))))
    print('Mean Specificity: %.3f%%' % (sum(specificity) / float(len(specificity))))
    print('Mean Accuracy: %.3f%%\n' % (sum(accuracy) / float(len(accuracy))))
    mean_sensitivity.append((sum(sensitivity) / float(len(sensitivity))))
    mean_specificity.append(sum(specificity) / float(len(specificity)))
    mean_accuracy.append(sum(accuracy) / float(len(accuracy)))

plt.plot(num_features, mean_accuracy, linestyle='-', linewidth=3, color='tab:blue', label='Accuracy')
plt.plot(num_features, mean_specificity, linestyle=':', linewidth=3, color='olivedrab', label="Specificity")
plt.plot(num_features, mean_sensitivity, linestyle='--', linewidth=3, color='crimson', label="Sensitivity")
plt.xlabel("Number of Features")
plt.ylabel("Percentage")
plt.title("Performance (5 trees)")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.grid(color='lightgrey')
plt.tight_layout()
plt.show()