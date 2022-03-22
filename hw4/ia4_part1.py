import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# plt.switch_backend('agg')

pd.set_option('display.max_columns', None)


class Node():
    """
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
    def __init__(
        self,
        prediction,
        feature,
        split,
        left_tree,
        right_tree,
    ):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


def entropy(target_col):
    """
    This function takes target_col, which is the data column containing the class labels, and returns H(Y).

    """
    M = len(target_col)

    edible = np.count_nonzero(target_col)
    poisonous = M - edible

    edible_pro = edible / M
    poisonous_pro = poisonous / M

    if edible_pro != 0 and poisonous_pro != 0:
        entropy = -1 * (edible_pro * math.log2(edible_pro) +
                        poisonous_pro * math.log2(poisonous_pro))
    else:
        entropy = 0

    return entropy


def InfoGain(data, split_attribute_name, target_name="class"):
    """
    This function calculates the information gain of specified feature. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """

    calculated_data = data[split_attribute_name].to_numpy()
    y = data[target_name].to_numpy()
    M = len(calculated_data)
    N = len(y)

    edible = np.count_nonzero(calculated_data)
    poisonous = M - edible
    edible_pro = edible / M
    poisonous_pro = poisonous / M
    edible_correct, poisonous_correct = 0, 0

    for i in range(0, N):
        if calculated_data[i] == 1 and y[i] == 1:
            edible_correct += 1
        elif calculated_data[i] == 0 and y[i] == 0:
            poisonous_correct += 1

    if (edible_correct != edible and edible_correct != 0):
        edible_entro = -1 * (
            (edible_correct / edible) * math.log2(edible_correct / edible) +
            (1 - edible_correct / edible) *
            math.log2(1 - edible_correct / edible))
    else:
        edible_entro = 0

    if (poisonous_correct != poisonous and poisonous_correct != 0):
        poisonous_entro = -1 * ((poisonous_correct / poisonous) *
                                math.log2(poisonous_correct / poisonous) +
                                (1 - poisonous_correct / poisonous) *
                                math.log2(1 - poisonous_correct / poisonous))
    else:
        poisonous_entro = 0

    Information_Gain = entropy(y) - (edible_pro * edible_entro +
                                     poisonous_pro * poisonous_entro)

    return Information_Gain


def DecisionTree(data, features, target_attribute_name, depth, maxdepth):
    """
    This function takes following paramters:
    1. data = the data for which the decision tree building algorithm should be run --> In the first run this equals the total dataset

    2. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset once we have splitted on a feature

    3. target_attribute_name = the name of the target attribute
    4. depth = the current depth of the node in the tree --- this is needed to remember where you are in the overall tree
    5. maxdepth =  the stopping condition for growing the tree

    """

    #First of all, define the stopping criteria here are some, depending on how you implement your code, there maybe more corner cases to consider
    """
    1. If max depth is met, return a leaf node labeled with majority class, additionally
    2. If all target_values have the same value (pure), return a leaf node labeled with majority class 
    3. If the remaining feature space is empty, return a leaf node labeled with majority class
    """
    y = data[target_attribute_name].to_numpy()
    M = len(y)
    infoGain_max = 0
    infoGain_sec = 0
    infoGain_third = 0
    best_feature = ""
    sec_feature = ""
    third_feature = ""

    if (depth >= maxdepth) or features == None:
        Leaf = Node(None, None, None, None, None)
        Leaf.prediction = np.bincount(y).argmax()
        return Leaf

    if (np.all(y == y[0])):
        Leaf = Node(None, None, None, None, None)
        Leaf.prediction = y[0]
        return Leaf

    #If none of the above holds true, grow the tree!
    #First, select the feature which best splits the dataset

    for feature in features:
        infoGain = InfoGain(data, feature)
        if infoGain > infoGain_max:
            infoGain_third = infoGain_sec
            infoGain_sec = infoGain_max
            infoGain_max = infoGain

            third_feature = sec_feature
            sec_feature = best_feature
            best_feature = feature

    #Once best split is decided, do the following:
    """
    1. create a node to store the selected feature 
    2. remove the selected feature from further consideration
    3. split the training data into the left and right branches and grow the left and right branch by making appropriate cursive calls
    4. return the completed node
    """

    root = Node(None, None, None, None, None)
    root.feature = best_feature
    print("Root of the tree:                      ", best_feature)
    print("Respective information gains of it:    ",
          InfoGain(data, best_feature))

    print("Second split of the tree:                      ", sec_feature)
    print("Respective information gains of it:    ", infoGain_sec)

    print("Third split of the tree:                      ", third_feature)
    print("Respective information gains of it:    ", infoGain_third)

    features.remove(best_feature)
    depth += 1

    left_tree_data = data[data[best_feature] == 1]
    right_tree_data = data[data[best_feature] == 0]

    root.left_tree = DecisionTree(left_tree_data, features, "class", depth,
                                  maxdepth)
    root.right_tree = DecisionTree(right_tree_data, features, "class", depth,
                                   maxdepth)

    return root


def search(data, features, tree):
    if tree.feature == None:
        return tree.prediction

    f_index = features.index(tree.feature)

    if data[f_index] == 1:
        return search(data, features, tree.left_tree)

    else:
        return search(data, features, tree.right_tree)


def predict(data, features, tree):
    Y = data["class"].to_numpy()
    M = len(Y)
    data = data.drop(columns="class")
    X = data.to_numpy()
    acc_num = 0

    for i in range(0, M):
        if search(X[i], features, tree) == Y[i]:
            acc_num += 1

    return (acc_num / len(Y))


def plot(ta, va):
    plt.plot(list(range(1, 11)), ta, label="Training Accuracy")
    plt.plot(list(range(1, 11)), va, label="Valdation Accuracy")
    plt.title("Accuracy of Average Perceptron")
    plt.xlabel("Dmax")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("part1_plot.png")
    plt.show()


def show(train_data, val_data):
    train_accuracy = []
    val_accuracy = []

    for dmax in range(1, 11):
        features = train_data.columns.tolist()
        features.remove("class")
        decision_tree = DecisionTree(train_data,
                                     features,
                                     "class",
                                     depth=0,
                                     maxdepth=dmax)
        features = train_data.columns.tolist()
        features.remove("class")
        train_accuracy.append(predict(train_data, features, decision_tree))
        val_accuracy.append((predict(val_data, features, decision_tree)))
    print("training accurecy:    ", train_accuracy)
    print("valdation accuracy:   ", val_accuracy)
    plot(train_accuracy, val_accuracy)


if __name__ == '__main__':

    train = pd.read_csv("train.csv")
    test = pd.read_csv("val.csv")
    features = train.columns.tolist()
    decision_tree = DecisionTree(train, features, "class", depth=0, maxdepth=2)

    show(train, test)
