from mysklearn.myclassifiers import MyRandomForrestClassifier

interview_table = [
    ["Senior", "Java", "no", "no", "False"],
    ["Senior", "Java", "no", "yes", "False"],
    ["Mid", "Python", "no", "no", "True"],
    ["Junior", "Python", "no", "no", "True"],
    ["Junior", "R", "yes", "no", "True"],
    ["Junior", "R", "yes", "yes", "False"],
    ["Mid", "R", "yes", "yes", "True"],
    ["Senior", "Python", "no", "no", "False"],
    ["Senior", "R", "yes", "no", "True"],
    ["Junior", "Python", "yes", "no", "True"],
    ["Senior", "Python", "yes", "yes", "True"],
    ["Mid", "Python", "no", "yes", "True"],
    ["Mid", "Java", "yes", "no", "True"],
    ["Junior", "Python", "no", "yes", "False"]
]

X_train = []
y = []
for row in interview_table:
    y.append(row[len(row) - 1])
    row.pop(len(row) - 1)
    X_train.append(row)
def test_my_random_forest_fit():
    forest_classifier = MyRandomForrestClassifier()
    forest_classifier.fit(X_train, y, seed=1)
    correct_trees = [['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 0, 2]], ['Value', 'Python', ['Leaf', 'True', 2, 6]], ['Value', 'R', ['Leaf', 'True', 2, 6]]], ['Attribute', 'att0', ['Value', 'no', ['Leaf', 'False', 1, 6]], ['Value', 'yes', ['Leaf', 'True', 5, 6]]], ['Attribute', 'att0', ['Value', 'no', ['Leaf', 'False', 1, 6]], ['Value', 'yes', ['Leaf', 'True', 5, 6]]]]
    print(forest_classifier.trees)
    assert forest_classifier.trees == correct_trees

def test_my_random_forest_predict():
    forest_classifier = MyRandomForrestClassifier()
    forest_classifier.fit(X_train, y, seed=1)
    assert forest_classifier.predict([["Mid", "Java", "yes", "no"], ["Junior", "Python", "no", "yes"]]) == ["True", "False"]