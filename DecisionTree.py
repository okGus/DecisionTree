# imports
import pandas as pd
import numpy as np
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Decision Tree functions creation
class Question:
  """
  Records a column number (e.g. 0 for Color) and a column value (e.g. Green)
  Match method is used to compare the feature value in an example to the 
  feature value stored in the question. See the demo
  """
  def __init__(self, column, value):
    self.column = column
    self.value = value

  def match(self, example):
    # Compare the feature value in an example to the
    # feature value in this question
    val = example[self.column]
    if is_numeric(val):
      return val >= self.value
    else:
      return val == self.value
  
  def __repr__(self):
    # helper method to print the question in a readable format
    condition = "=="
    if is_numeric(self.value):
      condition = ">="
    return f"Is {header[self.column]} {condition} {str(self.value)}?"
  
class Decision_Node:
  """A Decision Node asks a question
  
  This holds a reference to the question, and to the two child nodes.
  """

  def __init__(self, question, true_branch, false_branch):
    self.question = question
    self.true_branch = true_branch
    self.false_branch = false_branch

class Leaf:
  """A Leaf node classifies data
  
  This holds a dictionary of class (e.g. "Apple") -> number of times it
  appears in the rows from the training data that reach this leaf
  """

  def __init__(self, rows):
    self.predictions = class_counts(rows)

def class_counts(rows):
  """Counts the number of each type of example in a dataset"""
  counts = {}
  for label in rows.iloc[:, -1]: # last column
    # label is always the last column
    if label not in counts:
      counts[label] = 0
    counts[label] += 1
  return counts

def is_numeric(value):
  return isinstance(value, int) or isinstance(value, float)

def gini(rows):
  """Calculate the Gini Impurity for a list of rows
  """
  counts = class_counts(rows)
  impurity = 1
  for lbl in counts:
    prob_of_lbl = counts[lbl] / float(len(rows))
    impurity -= prob_of_lbl**2
  return impurity

def entropy(rows):
  counts = class_counts(rows)
  en = 0
  for lbl in counts:
    en += counts[lbl]/float(len(rows)) * log2(counts[lbl]/float(len(rows)))
  en = -(en)
  return en

def en_info_gain(left, right, current_uncertainty):
  gain = 0
  left = pd.DataFrame(left)
  right = pd.DataFrame(right)
  weight = ( len(left) / 10**len(str(len(left))) ) * entropy(left) + ( len(right) / 10**len(str(len(right))) ) * entropy(right)
  en_gain = current_uncertainty - weight
  return en_gain

def partition(rows, question):
  """Partitions a dataset

  For each row in the dataset, check if it matches the question. If so
  add it to 'true rows', otherwise, add it to 'false rows'
  """

  true_rows, false_rows = [], []
  row_length = rows.shape[0]
  for row in range(row_length):
    m_row = rows.iloc[row].tolist()
    if question.match(m_row):
      true_rows.append(m_row)
    else:
      false_rows.append(m_row)
  return true_rows, false_rows

def info_gain(left, right, current_uncertainty):
  """Information Gain
  
  The uncertainty of the starting node, minus the weighted 
  impurity of the two child nodes
  """

  p = float(len(left)) / (len(left) + len(right))
  return current_uncertainty - p * gini(left) - (1-p) * gini(right)

def find_best_split(rows):
  """Find the best question to ask by iterating over every feature / value
  and calculating the infomation gain
  """
  #ipdb.set_trace() 
  best_gain = 0 # keep track of the best information gain
  best_question = None # keep track of the feature / value that produced it
  current_uncertainty = entropy(rows)
  n_features = rows.shape[1] - 1 # numer of columns

  for col in range(n_features): # for each feature
    values = set(rows[col].unique().tolist()) # unique values in the column
    for val in values: # for each value
      question = Question(col, val)
      
      # try splitting the dataset
      true_rows, false_rows = partition(rows, question)

      # skip this split if it doesn't divide the dataset
      if len(true_rows) == 0 or len(false_rows) == 0:
        continue

      # calculate the information gain from this split
      gain = en_info_gain(true_rows, false_rows, current_uncertainty)

      if gain >= best_gain:
        best_gain, best_question = gain, question

  return best_gain, best_question

def build_tree(rows):
  """Builds the tree
  """
  gain, question = find_best_split(rows)

  if gain == 0:
    return Leaf(rows)

  true_rows, false_rows = partition(rows, question)
  true_rows = pd.DataFrame(true_rows)
  false_rows = pd.DataFrame(false_rows)

  true_branch = build_tree(true_rows)

  false_branch = build_tree(false_rows)

  return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
  # Base case: we've reached a leaf
  if isinstance(node, Leaf):
    print(spacing + "Predict", node.predictions)
    return

  # Print the question at this node
  print(spacing + str(node.question))

  # Call this funcitno recursively on the true branch
  print(spacing + '--> True:')
  print_tree(node.true_branch, spacing + " ")

  # Call this funcitno recursively on the false branch
  print(spacing + '--> False:')
  print_tree(node.false_branch, spacing + " ")

def print_leaf(counts):
  """A nicer way to print the predictions at a leaf."""
  total = sum(counts.values()) * 1.0
  probs = {}
  for lbl in counts.keys():
    probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
  return probs

def classify(row, node):
  # Base case: we've reach a leaf
  if isinstance(node, Leaf):
    return node.predictions
  
  # Decide whether to follow the true-branch or the false-branch
  # compare the feature / value stored in the node,
  # to the example were considering.

  if node.question.match(row):
    return classify(row, node.true_branch)
  else:
    return classify(row, node.false_branch)

def predict(row, node):
  """Yes its the same as classify, so?"""
  # Base case: we've reach a leaf
  if isinstance(node, Leaf):
    prediction = node.predictions
    final_prediction = sorted(prediction.items(), key=lambda v: v[1], reverse=True)
    return final_prediction[0][0]
  
  # Decide whether to follow the true-branch or the false-branch
  # compare the feature / value stored in the node,
  # to the example were considering.

  if node.question.match(row):
    return predict(row, node.true_branch)
  else:
    return predict(row, node.false_branch)

def accuracy(y_true, y_pred):
  """Just to see if it works like sklearn.metrics accuracy_score"""
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy
  
  
#### Main #####
header = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety']

# Download from UCI ML Repo
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data

df = pd.read_csv('./car.data', header=None)
df.columns = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Label']
df.head()

feature_cols = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety']
X = df[feature_cols]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=None)
train_data = pd.merge(X_train, y_train, left_index=True, right_index=True)
train_data.columns = [0, 1, 2, 3, 4, 5, 6]
my_tree = build_tree(train_data)
print_tree(my_tree)

test_data = pd.merge(X_test, y_test, left_index=True, right_index=True)
test_data.columns = [0, 1, 2, 3, 4, 5, 6]
test_data.head()

y_predict = []
for idx in range(len(test_data)):
  row = list(test_data.iloc[idx][:-1])
  predictions = predict(row, my_tree)
  #predictions = list(predictions.keys())[0]
  y_predict.append(predictions)
  
acc = accuracy(y_test, y_predict)
print('Accuracy: ', acc)
