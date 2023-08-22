# Bibliotecas de manipualção e visualização de dados
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.io import arff
from sklearn.cluster import AgglomerativeClustering


#Funções de avaliação dos modelos
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, 
                             ConfusionMatrixDisplay, 
                             roc_curve,RocCurveDisplay,
                             f1_score)
from sklearn.model_selection import (train_test_split, 
                                     KFold, 
                                     LeaveOneOut, 
                                     StratifiedKFold, 
                                     GridSearchCV)

# Classes do modelo de aprendizado
from sklearn.naive_bayes import (GaussianNB, 
                                 BernoulliNB, 
                                 MultinomialNB)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import (DecisionTreeClassifier, 
                          DecisionTreeRegressor, 
                          plot_tree)
from sklearn.svm import SVC

# Lib de Warnings
import warnings
warnings.filterwarnings('ignore')

#==========FIM DO BLOCO DE IMPORTAÇÕES===========================
#==========DECLARAÇÃO DE VARIÁVEIS===============================

#declaração da variável com folder e arquivo do DATASET
folder_dataset = ("C:\\Users\\rotto\\OneDrive\\MACHINE LEARNING\\OneDrive_2023-08-19\\"
    "00 - Machine Learning\\ML 2\\TCC\\ML2\\PROJETO\\PYTHON\\DATASET\\dataset_6_letter.arff")

#==========FIM DO BLOCO DE DECLARAÇÃO DE VARIÁVEIS================

#leitura do DATASET e respectivo print em tela
dataset, meta = arff.loadarff(folder_dataset)
dataset = pd.DataFrame(dataset)
dataset.head()
print(dataset)

# Distribuição das classes por Features
target_col = 'class'
print("Quantas classes existem nesse dataset?\n%d" %(len(dataset[target_col].unique())))
print("\nQuantas instâncias existem no dataset?\n%d" %(dataset.shape[0]))
print("\nQuantas features existem no dataset?\n%d" % (dataset.shape[1]-1))
print("\nQue features são essas?\n%s" % (str([k for k in dataset.keys() if k != target_col])))
print("\nQual o numero de instâncias por classe?")
print(dataset[target_col].value_counts())


sns.set_style("whitegrid")
sns.FacetGrid(dataset, hue =target_col,
              height = 8).map(plt.scatter,
                              'width',
                              'high').add_legend()