
#import das bibliotecas do projeto
import numpy as np 
import pandas as pd
from scipy.io import arff

#declaração da variável com folder e arquivo do DATASET
folder_dataset = ("C:\\Users\\rotto\\OneDrive\\MACHINE LEARNING\\OneDrive_2023-08-19\\"
    "00 - Machine Learning\\ML 2\\TCC\\ML2\\PROJETO\DATASET\\dataset_6_letter.arff")

#leitura do DATASET e respectivo print em tela
dataframe, meta = arff.loadarff(folder_dataset)
dataframe = pd.DataFrame(dataframe)
dataframe.head()
print(dataframe)


