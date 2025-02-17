
import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('mnist_train.csv')
    test_df = pd.read_csv('mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    x_tr_min = X_train.min()
    x_tr_max = X_train.max()
    X_train = 2*((X_train-x_tr_min)/(x_tr_max-x_tr_min)) - 1
    x_te_min = X_test.min()
    x_te_max = X_test.max()
    X_test = 2*((X_test-x_te_min)/(x_te_max-x_te_min)) - 1
    return X_train,X_test
    raise NotImplementedError


def plot_metrics(metrics) -> None:
    k = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    for i in range(len(metrics)):
        k.append(metrics[i][0])
        accuracy.append(metrics[i][1])
        precision.append(metrics[i][2])
        recall.append(metrics[i][3])
        f1_score.append(metrics[i][4])
    plt.plot(k,accuracy)
    plt.xlabel("Components")
    plt.ylabel("Accuracy")
    plt.title("Components vs Accuracy")
    plt.savefig("k_accuracy.jpg")
    plt.show()
    plt.plot(k,precision)
    plt.xlabel("Components")
    plt.ylabel("Precision")
    plt.title("Components vs Precision")
    plt.savefig("k_precision.jpg")
    plt.show()
    plt.plot(k,recall)
    plt.xlabel("Components")
    plt.ylabel("Recall")
    plt.title("components vs Recall")
    plt.savefig("k_recall.jpg")
    plt.show()
    plt.plot(k,f1_score)
    plt.xlabel("Components")
    plt.ylabel("F1_score")
    plt.title("Components vs F1_score")
    plt.savefig("k_f1_score.jpg")
    plt.show()
    # raise NotImplementedError
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        covar = np.cov(X.T)
        Eval,Evac = np.linalg.eigh(covar)
        k_Evac = Evac[:,-self.n_components:]
        self.components = k_Evac
        # return k_Evac
        # raise NotImplementedError
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        return np.matmul(self.components.T,X.T).T
        raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)
    
class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        n,d = X.shape
        self.w = np.zeros(d)
        self.b = 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            n,d = X.shape
            # for j in range(0,n):
            index = np.random.choice(n)
            xj,yj = X[index],y[index]
            if(yj*(np.dot(self.w,xj)+ self.b) < 1):
                self.w = self.w + learning_rate*(C*yj*xj - self.w)
                self.b = self.b + learning_rate * C*yj    # update in else?
            # raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.matmul(X,self.w) + self.b   # add np.sign
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        C,learning_rate,num_iters = kwargs.values()
        for i in range(len(self.models)):
            y_new = np.where(y==i,1,-1)
            self.models[i].fit(X,y_new,learning_rate,num_iters,C)
        # then train the 10 SVM models using the preprocessed data for each class

        # raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        pred = np.zeros((X.shape[0],self.num_classes))
        for i in range(len(self.models)):
            pred[:,i] = self.models[i].predict(X)
        return (np.argmax(pred,axis=1))
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        precision = 0
        for i in range(len(np.unique(y))):
            t_p = np.sum(((y == i) & (y_pred == i)))
            f_p = np.sum(((y != i) & (y_pred == i)))
            if(t_p + f_p == 0):
                precision += 0
            else:
                precision += t_p / (t_p+f_p)
        return precision / len(np.unique(y))
        raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        recall = 0
        for i in range(len(np.unique(y))):
            t_p = np.sum(((y == i) & (y_pred == i)))
            f_n = np.sum(((y == i) & (y_pred != i)))
            if(t_p + f_n == 0):
                recall += 0
            else:
                recall += t_p / (t_p+f_n)
        return recall / len(np.unique(y))
        raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        pre_score = self.precision_score(X,y)
        rec_score = self.recall_score(X,y)
        return 2*((pre_score*rec_score)/(pre_score+rec_score))
        raise NotImplementedError
    
def get_hyperparameters() -> Tuple[float, int, float]:
    # get the hyperparameters
    learning_rate = 0.0001  #0.001 less acc(84%)
    num_iters = 270000  # 240000
    C = 10  #10
    return learning_rate,num_iters,C
    raise NotImplementedError


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # normalize the data
    X_train, X_test = normalize(X_train, X_test)

    metrics = []
    for k in [5, 10, 20, 50, 100, 200, 500]:    # 
        # reduce the dimensionality of the data
        pca = PCA(n_components=k)
        X_train_emb = pca.fit_transform(X_train)
        X_test_emb = pca.transform(X_test)

        # create a model
        svm = MultiClassSVM(num_classes=10)

        # fit the model
        svm.fit(
            X_train_emb, y_train, C=C,
            learning_rate=learning_rate,
            num_iters=num_iters,
        )

        # evaluate the model
        accuracy = svm.accuracy_score(X_test_emb, y_test)
        precision = svm.precision_score(X_test_emb, y_test)
        recall = svm.recall_score(X_test_emb, y_test)
        f1_score = svm.f1_score(X_test_emb, y_test)

        metrics.append((k, accuracy, precision, recall, f1_score))

        print(f'k={k}, accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')

    # plot and save the results
    plot_metrics(metrics)


if __name__ == '__main__':
    main()
