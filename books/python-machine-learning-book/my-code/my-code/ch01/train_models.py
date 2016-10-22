import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed

from perceptron import Perceptron
from adaptive_linear_neuron import AdalineGD
from adaptive_online_learning import AdalineSGD

def load_data():
	"""Load iris dataset"""
	df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data', header=None)
	# select setosa and versicolor
	y = df.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)

	# extract sepal length and petal length
	X = df.iloc[0:100, [0, 2]].values
	return X,y

def plot_iris_data(X, y):
	"""Plotting the Iris data"""
	# plot data
	plt.scatter(X[:50, 0], X[:50, 1],
	            color='red', marker='o', label='setosa')
	plt.scatter(X[50:100, 0], X[50:100, 1],
	            color='blue', marker='x', label='versicolor')

	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.legend(loc='upper left')

	plt.tight_layout()
	#plt.savefig('./images/02_06.png', dpi=300)
	plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

def train_perceptron(X, y):
	"""Training the perceptron model"""
	ppn = Perceptron(eta=0.1, n_iter=10)

	ppn.fit(X, y)

	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Number of updates')

	plt.tight_layout()
	# plt.savefig('./perceptron_1.png', dpi=300)
	plt.show()

	plot_decision_regions(X, y, classifier=ppn)
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.legend(loc='upper left')

	plt.tight_layout()
	# plt.savefig('./perceptron_2.png', dpi=300)
	plt.show()

def train_adaline_raw(X, y):
	"""adaptive linear neuron in raw features"""
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

	ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
	ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('log(Sum-squared-error)')
	ax[0].set_title('Adaline - Learning rate 0.01')

	ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
	ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('Sum-squared-error')
	ax[1].set_title('Adaline - Learning rate 0.0001')

	plt.tight_layout()
	# plt.savefig('./adaline_1.png', dpi=300)
	plt.show()

def standardize_features(X):
	# standardize features
	X_std = np.copy(X)
	X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
	X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
	return X_std

def train_adaline(X_std, y):
	"""Adaptive linear neuron in standardize features"""
	ada = AdalineGD(n_iter=15, eta=0.01)
	ada.fit(X_std, y)

	plot_decision_regions(X_std, y, classifier=ada)
	plt.title('Adaline - Gradient Descent')
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.legend(loc='upper left')
	plt.tight_layout()
	# plt.savefig('./adaline_2.png', dpi=300)
	plt.show()

	plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Sum-squared-error')

	plt.tight_layout()
	# plt.savefig('./adaline_3.png', dpi=300)
	plt.show()

def train_adaline_sgd(X_std, y):
	"""Adaptive linear neuron with Stochastic Gradient Descent in standardize features"""
	ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
	ada.fit(X_std, y)

	plot_decision_regions(X_std, y, classifier=ada)
	plt.title('Adaline - Stochastic Gradient Descent')
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.legend(loc='upper left')

	plt.tight_layout()
	#plt.savefig('./adaline_4.png', dpi=300)
	plt.show()

	plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Average Cost')

	plt.tight_layout()
	# plt.savefig('./adaline_5.png', dpi=300)
	plt.show()

def main():
	X,y = load_data()

	# plot_iris_data(X, y)

	# # Perceptron
	# train_perceptron(X, y)

	# # Adaline with Gradient Descent
	# train_adaline_raw(X, y)
	X_std = standardize_features(X)
	# train_adaline(X_std, y)

	# Adaline with Stochastic Gradient Descent
	train_adaline_sgd(X_std, y)

if __name__ == '__main__':
    main()