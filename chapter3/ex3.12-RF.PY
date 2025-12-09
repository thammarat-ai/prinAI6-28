# Example 3.12 Random Forest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset and split into training and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Create and train the Random Forest classifier
clf = RandomForestClassifier()

#
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate and print the number of correctly labeled points
print(f"Total points: {y_test.shape[0]} Correctly labeled points : {(y_test == y_pred).sum()}")
