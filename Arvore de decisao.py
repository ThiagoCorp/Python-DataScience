#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 08:59:02 2018

@author: thiago
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

dados = pd.read_csv("Credit.csv")
previsores = dados.iloc[:,0:20].values
classe = dados.iloc[:,20].values

labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:, 6])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder.fit_transform(previsores[:, 9])
previsores[:, 11] = labelencoder.fit_transform(previsores[:, 11])
previsores[:, 13] = labelencoder.fit_transform(previsores[:, 13])
previsores[:, 14] = labelencoder.fit_transform(previsores[:, 14])
previsores[:, 16] = labelencoder.fit_transform(previsores[:, 16])
previsores[:, 18] = labelencoder.fit_transform(previsores[:, 18])
previsores[:, 19] = labelencoder.fit_transform(previsores[:, 19])

X_treinamento, X_teste, y_treinamento, y_teste =  train_test_split(previsores, classe,test_size = 0.3,random_state = 0)

arvore =  DecisionTreeClassifier()
arvore.fit(X_treinamento,y_treinamento)
export_graphviz(arvore,out_file = "tree.dot")
analise = arvore.predict(X_teste)
matrix = confusion_matrix(y_teste,analise)
taxaacerto = accuracy_score(y_teste,analise)
taxaerro = 1 - taxaacerto
print("A taxa de acerto foi de:%5.2f"%taxaacerto)
print("A taxa de erro foi de:%5.2f"%taxaerro)
