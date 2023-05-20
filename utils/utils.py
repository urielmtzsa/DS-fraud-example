import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_val_score
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix, f1_score,roc_curve
from scipy.stats import ks_2samp
import graphviz

def completitud(df):
    
    """
    Devuelve un dataframe que te dice por campo: la completitud (porcentaje de registros
    que no son missings), el número de missings, el tipo de dato
    
    
    Parameters
    ----------   
    df : dataframe
        Dataframe a ser evaluado
        
    Returns
    -------
    df
        Dataframe con 4 columnas: nombre del campo, completitud, missings, type
    """   
    
    completitud=pd.DataFrame((1-df.isnull().sum()/df.shape[0])*100).reset_index().rename(columns={"index":"df",0:"completitud"})
    missings=pd.DataFrame(df.isnull().sum()).reset_index().rename(columns={"index":"df",0:"missings"})
    type=pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"df",0:"type"})
    var=pd.DataFrame(df.var()).reset_index().rename(columns={"index":"df",0:"varianza"})
    completitud=completitud.merge(missings,how="outer",on=["df"])
    completitud=completitud.merge(type,how="outer",on=["df"])
    completitud=completitud.merge(var,how="outer",on=["df"])

    return completitud


def supervisado_clasificacion(X_entrenamiento,y_entrenamiento,X_prueba,y_prueba,semilla,folds,iteraciones,score,param_grid):
    """
    Entrena un modelo tipo regresión logística. Calibra los parámetros
    del modelo a través de RandomizedSearchCV. 
    
    Parameters
    ----------   
    X_entrenamiento : dataframe
        Dataframe que contiene las variables explicativas para el conjunto de entrenamiento
    y_entrenamiento : dataframe
        Dataframe que contiene la variable target para el conjunto de entrenamiento
    X_prueba: dataframe
        Dataframe que contiene las variables explicativas para el conjunto de prueba
    y_prueba: dataframe
        Dataframe que contiene la variable target para el conjunto de prueba
    semilla : float
        Random state para hacer el proceso replicable
    folds : int
        Folds para RandomizedSearchCV, la búsqueda de los mejores hiperparámetros
    iteraciones : floar
        Valor entre 0 y 1. Porcentaje de iteraciones para RandomizedSearchCV que depende
        de la cardinalidad del espacio de hiperparámetros en el que se busca en cada modelo
    score : str
        Score con el que se evalúa al mejor modelo    
    param_grid: dict
        Espacio de búsqueda de hiperparámetros
    Returns
    -------
    dict
        Diccionario con: objeto del RandomizedSearchCV en el que se encuentra el mejor
        modelo, objeto del mejor modelo, score del mejor modelo
    """   
    
    modelos={}
    tabla_scores=pd.DataFrame(columns=["modelo","score_searched"])
        
    #param_grid_logistica = {"penalty": ['l1', 'l2', 'elasticnet', 'none'] ,
    #                        "tol": [x/2500 for x in range(26)],
    #                        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #                        "random_state": [semilla]}
    
    regresion_logistica = LogisticRegression()
    regresion_logistica.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    combinaciones=np.prod(list(map(len, param_grid.values())))
    clf = RandomizedSearchCV(estimator=regresion_logistica, param_distributions=param_grid, 
                             cv=folds, scoring=score, n_iter=int(combinaciones*iteraciones)+1,
                             error_score=-1000, n_jobs=-1, verbose=5, random_state=semilla)
    clf.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])    
    tabla_scores.loc[len(tabla_scores)] =["regresion_logistica",clf.best_score_]
    modelos["regresion_logistica"]={ "modelo":clf,
                                     "parametros":clf.best_estimator_,
                                     "roc":clf.best_score_}    
    
    
    tabla_scores=tabla_scores.sort_values(by="score_searched",ascending=False)
    modelos["tabla_scores"]=tabla_scores
    
    
    return(modelos)



def supervisado_clasificacion_scores(X_entrenamiento,y_entrenamiento,X_prueba,y_prueba,semilla,folds,score,
                                     parameters
                                     ):
    
    """
    Aplica para un modelo tipo regresión logística. Realiza Cross Validation. Reporta 
    roc-auc para test, la media y desviación estándar de los scores en Cross Validation, 
    accuracy para test, F1-score para test, matriz de confusión para test.    
    Parameters
    ----------   
    X_entrenamiento : dataframe
        Dataframe que contiene las variables explicativas para el conjunto de entrenamiento
    y_entrenamiento : dataframe
        Dataframe que contiene la variable target para el conjunto de entrenamiento
    X_prueba: dataframe
        Dataframe que contiene las variables explicativas para el conjunto de prueba
    y_prueba: dataframe
        Dataframe que contiene la variable target para el conjunto de prueba
    semilla : float
        Random state para hacer el proceso replicable
    folds : int
        Folds para CrossValidation
    score : str
        Score con el que se evalúa al mejor modelo  
    parameters: dict
        Diccionario con mejores hiperparámetros
    Returns
    -------
    dataframe
        Tabla con resultados del modelo
    """   
    
    tabla_scores=pd.DataFrame(columns=["modelo","roc","score_mean_cross_val","score_std_cross_val","accuracy","F1","confusion","ks","thr"])
    
    
    regresion_logistica = LogisticRegression(**parameters)
    regresion_logistica.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    ls_medias = cross_val_score(estimator=regresion_logistica, X=X_entrenamiento, y = y_entrenamiento[y_entrenamiento.columns[0]], cv = folds, n_jobs=-1, scoring=score)
    
    fpr, tpr, thresholds = roc_curve(y_entrenamiento,regresion_logistica.predict_proba(X_entrenamiento)[:,1])
    Gmeans = tpr * (1-fpr)
    thr = np.argmax(Gmeans)
    thr = thresholds[thr]
    
    
    tabla_scores.loc[len(tabla_scores)] =["regresion_logistica",
                                          roc_auc_score(y_score=regresion_logistica.predict_proba(X_prueba)[:,1],y_true=y_prueba) if score=="roc_auc" else 0,            
                                          np.mean(ls_medias), 
                                          np.std(ls_medias),
                                          accuracy_score(y_pred=regresion_logistica.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          f1_score(y_pred=regresion_logistica.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba,average="binary" if score=="roc_auc" else 'weighted'),
                                          confusion_matrix(y_pred=regresion_logistica.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          ks_2samp(regresion_logistica.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==0],regresion_logistica.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==1])[0],
                                          thr
                                         ]    


    tabla_scores=tabla_scores.sort_values(by="roc",ascending=False)
    return(tabla_scores)



def supervisado_clasificacion_DecisionTree(X_entrenamiento,y_entrenamiento,X_prueba,y_prueba,semilla,
                                           folds_search,folds_CV,iteraciones,score,param_grid_tree,param_grid_forest):
    
    """
    Entrena un modelo de árbol de decisión y un modelo randon forest. Calibra los parámetros
    del modelo a través de RandomizedSearchCV. Realiza Cross Validation. Reporta el mejor 
    score para el conjunto de entrenamiento, roc-auc para test, la media y desviación estándar 
    de los scores en Cross Validation, accuracy para test, F1-score para test, matriz de 
    confusión para test.    
    Parameters
    ----------   
    X_entrenamiento : dataframe
        Dataframe que contiene las variables explicativas para el conjunto de entrenamiento
    y_entrenamiento : dataframe
        Dataframe que contiene la variable target para el conjunto de entrenamiento
    X_prueba: dataframe
        Dataframe que contiene las variables explicativas para el conjunto de prueba
    y_prueba: dataframe
        Dataframe que contiene la variable target para el conjunto de prueba
    semilla : float
        Random state para hacer el proceso replicable
    folds_search : int
        Folds para RandomizedSearchCV, la búsqueda de los mejores hiperparámetros
    folds_CV : int
        Folds para CrossValidation después de encontrar los mejores hiperparámetros
    iteraciones : floar
        Valor entre 0 y 1. Porcentaje de iteraciones para RandomizedSearchCV que depende
        de la cardinalidad del espacio de hiperparámetros en el que se busca en cada modelo
    score : str
        Score con el que se evalúa al mejor modelo  
    param_grid_tree: dict
        Espacio de búsqueda de hiperparámetros para DecisionTreeClassifier
    param_grid_forest: dict
        Espacio de búsqueda de hiperparámetros para RandomForestClassifier
    Returns
    -------
    dict
        Diccionario con: objeto del RandomizedSearchCV en el que se encuentra el mejor
        modelo, objeto del mejor modelo, dataframe con resultados del modelo, y para 
        decision tree gráfico del árbol de decisión
    """
    modelos={}
    tabla_scores=pd.DataFrame(columns=["modelo","score_searched","roc","score_mean_cross_val","score_std_cross_val","accuracy","F1","confusion","ks","thr"])
 

    #param_grid_tree = {"criterion" : ["gini", "entropy"],
    #                   "max_depth" : [x for x in range(3,11)],
    #                   "min_samples_split": [x/100 for x in range(1,11)],
    #                   "min_samples_leaf": [x/100 for x in range(1,11)],
    #                   "max_features" : [None,"auto", "sqrt", "log2"],
    #                   "random_state": [semilla]
    #                   }
    
    #param_grid_forest = {"criterion" : ["gini", "entropy"],
    #                   "max_depth" : [x for x in range(3,11)],
    #                   "min_samples_split": [x*10 for x in range(1,11)],
    #                   "min_samples_leaf": [x*10 for x in range(1,11)],
    #                   "max_features" : [None,"auto", "sqrt", "log2"],
    #                   "random_state": [semilla]
    #                   }
      
    tree = DecisionTreeClassifier()
    tree.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    combinaciones=np.prod(list(map(len, param_grid_tree.values())))
    clf = RandomizedSearchCV(estimator=tree, param_distributions=param_grid_tree, cv=folds_search, scoring=score, 
                       error_score=-1000, n_jobs=-1, verbose=5,
                            n_iter=int(combinaciones*iteraciones)+1, random_state=semilla)
    clf.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])  
    tree_ajustado = DecisionTreeClassifier(**clf.best_params_)
    tree_ajustado.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    ls_medias = cross_val_score(estimator=tree_ajustado, X=X_entrenamiento, 
                                y = y_entrenamiento[y_entrenamiento.columns[0]], cv = folds_CV, n_jobs=-1, 
                                scoring=score)
    
    fpr, tpr, thresholds = roc_curve(y_entrenamiento,tree_ajustado.predict_proba(X_entrenamiento)[:,1])
    Gmeans = tpr * (1-fpr)
    thr = np.argmax(Gmeans)
    thr = thresholds[thr]
    
    
    tabla_scores.loc[len(tabla_scores)] =["tree_classifier",
                                          clf.best_score_,
                                          roc_auc_score(y_score=tree_ajustado.predict_proba(X_prueba)[:,1],y_true=y_prueba) if score=="roc_auc" else 0,            
                                          np.mean(ls_medias), 
                                          np.std(ls_medias),
                                          accuracy_score(y_pred=tree_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          f1_score(y_pred=tree_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba,average="binary" if score=="roc_auc" else 'weighted'),
                                          confusion_matrix(y_pred=tree_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          ks_2samp(tree_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==0],tree_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==1])[0],
                                          thr
                                         ]
    dot_data = export_graphviz(tree_ajustado, out_file=None, 
                               class_names=[str(x) for x in list(y_entrenamiento[y_entrenamiento.columns[0]].unique())],
                               feature_names=list(X_entrenamiento.columns),  
                               filled=True)
    graph=graphviz.Source(dot_data, format="png") 
    modelos["tree_classifier"]={"modelo":clf,
                                "parametros":clf.best_estimator_,
                                "tree":graph}
    
    
    
    forest = RandomForestClassifier()
    forest.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    combinaciones=np.prod(list(map(len, param_grid_forest.values())))
    clf = RandomizedSearchCV(estimator=forest, param_distributions=param_grid_forest, cv=folds_search, scoring=score, 
                       error_score=-1000, n_jobs=-1, verbose=5,
                            n_iter=int(combinaciones*iteraciones)+1, random_state=semilla)
    clf.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])  
    forest_ajustado = RandomForestClassifier(**clf.best_params_)
    forest_ajustado.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    ls_medias = cross_val_score(estimator=forest_ajustado, X=X_entrenamiento, 
                                y = y_entrenamiento[y_entrenamiento.columns[0]], cv = folds_CV, n_jobs=-1, 
                                scoring=score)
    
    fpr, tpr, thresholds = roc_curve(y_entrenamiento,forest_ajustado.predict_proba(X_entrenamiento)[:,1])
    Gmeans = tpr * (1-fpr)
    thr = np.argmax(Gmeans)
    thr = thresholds[thr]
    
    
    tabla_scores.loc[len(tabla_scores)] =["forest_classifier",
                                          clf.best_score_,
                                          roc_auc_score(y_score=forest_ajustado.predict_proba(X_prueba)[:,1],y_true=y_prueba) if score=="roc_auc" else 0,            
                                          np.mean(ls_medias), 
                                          np.std(ls_medias),
                                          accuracy_score(y_pred=forest_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          f1_score(y_pred=forest_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba,average="binary" if score=="roc_auc" else 'weighted'),
                                          confusion_matrix(y_pred=forest_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          ks_2samp(forest_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==0],forest_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==1])[0],
                                          thr
                                         ]
    modelos["forest_classifier"]={"modelo":clf,
                                "parametros":clf.best_estimator_}

    
    tabla_scores=tabla_scores.sort_values(by="score_searched",ascending=False)
    modelos["tabla_scores"]=tabla_scores
    
    
    return(modelos)



def supervisado_clasificacion_redes_neuronales(X_entrenamiento,y_entrenamiento,X_prueba,y_prueba,semilla,folds_search,folds_CV,iteraciones,score,param_grid):        
    """
    Entrena una red neuronal de scikit-learn. Calibra los parámetros
    del modelo a través de RandomizedSearchCV. Realiza Cross Validation. Reporta el mejor 
    score para el conjunto de entrenamiento, roc-auc para test, la media y desviación estándar 
    de los scores en Cross Validation, accuracy para test, F1-score para test, matriz de 
    confusión para test.    
    Parameters
    ----------   
    X_entrenamiento : dataframe
        Dataframe que contiene las variables explicativas para el conjunto de entrenamiento
    y_entrenamiento : dataframe
        Dataframe que contiene la variable target para el conjunto de entrenamiento
    X_prueba: dataframe
        Dataframe que contiene las variables explicativas para el conjunto de prueba
    y_prueba: dataframe
        Dataframe que contiene la variable target para el conjunto de prueba
    semilla : float
        Random state para hacer el proceso replicable
    folds_search : int
        Folds para RandomizedSearchCV, la búsqueda de los mejores hiperparámetros
    folds_CV : int
        Folds para CrossValidation después de encontrar los mejores hiperparámetros
    iteraciones : floar
        Valor entre 0 y 1. Porcentaje de iteraciones para RandomizedSearchCV que depende
        de la cardinalidad del espacio de hiperparámetros en el que se busca en cada modelo
    score : str
        Score con el que se evalúa al mejor modelo  
    param_grid: dict
        Espacio de búsqueda de hiperparámetros
    Returns
    -------
    dict
        Diccionario con: objeto del RandomizedSearchCV en el que se encuentra el mejor
        modelo, objeto del mejor modelo, dataframe con resultados del modelo
    """    
    modelos={}
    tabla_scores=pd.DataFrame(columns=["modelo","score_searched","roc","score_mean_cross_val","score_std_cross_val","accuracy","F1","confusion","ks","thr"])    
    
    #param_grid_mlp = {   "max_iter": [50,100,150,200],        
    #                     'hidden_layer_sizes': [(50,50,50), (100,50,50), (50,100,50), (50,50,100), (100,)],
    #                     'activation': ['tanh', 'relu',"logistic"],
    #                     'solver': ['sgd', 'adam'],
    #                     'alpha': [0.00001,0.0001,0.001],
    #                     'learning_rate': ['constant','adaptive',"invscaling"],
    #                     "random_state": [semilla]
    #               }
      
    mlp = MLPClassifier()
    mlp.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    combinaciones=np.prod(list(map(len, param_grid.values())))
    clf = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid, cv=folds_search, scoring=score, 
                       error_score=-1000, n_jobs=-1, verbose=5,
                            n_iter=int(combinaciones*iteraciones)+1, random_state=semilla)
    clf.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])  
    mlp_ajustado = MLPClassifier(**clf.best_params_)
    mlp_ajustado.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    ls_medias = cross_val_score(estimator=mlp_ajustado, X=X_entrenamiento, y = y_entrenamiento[y_entrenamiento.columns[0]], cv = folds_CV, n_jobs=-1, scoring=score)
    
    fpr, tpr, thresholds = roc_curve(y_entrenamiento,mlp_ajustado.predict_proba(X_entrenamiento)[:,1])
    Gmeans = tpr * (1-fpr)
    thr = np.argmax(Gmeans)
    thr = thresholds[thr]
    
    tabla_scores.loc[len(tabla_scores)] =["mlp_classifier",
                                          clf.best_score_,
                                          roc_auc_score(y_score=mlp_ajustado.predict_proba(X_prueba)[:,1],y_true=y_prueba) if score=="roc_auc" else 0,            
                                          np.mean(ls_medias), 
                                          np.std(ls_medias),
                                          accuracy_score(y_pred=mlp_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          f1_score(y_pred=mlp_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba,average="binary" if score=="roc_auc" else 'weighted'),
                                          confusion_matrix(y_pred=mlp_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          ks_2samp(mlp_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==0],mlp_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==1])[0],
                                          thr
                                         ]
    modelos["mlp_classifier"]={"modelo":clf,
                    "parametros":clf.best_estimator_}
    
    tabla_scores=tabla_scores.sort_values(by="score_searched",ascending=False)
    modelos["tabla_scores"]=tabla_scores
    
    
    return(modelos)



def supervisado_clasificacion_xgb(X_entrenamiento,y_entrenamiento,X_prueba,y_prueba,semilla,folds_search,folds_CV,iteraciones,score,param_grid):
    
    """
    Entrena XGBoost. Calibra los parámetros
    del modelo a través de RandomizedSearchCV. Realiza Cross Validation. Reporta el mejor 
    score para el conjunto de entrenamiento, roc-auc para test, la media y desviación estándar 
    de los scores en Cross Validation, accuracy para test, F1-score para test, matriz de 
    confusión para test.    
    Parameters
    ----------   
    X_entrenamiento : dataframe
        Dataframe que contiene las variables explicativas para el conjunto de entrenamiento
    y_entrenamiento : dataframe
        Dataframe que contiene la variable target para el conjunto de entrenamiento
    X_prueba: dataframe
        Dataframe que contiene las variables explicativas para el conjunto de prueba
    y_prueba: dataframe
        Dataframe que contiene la variable target para el conjunto de prueba
    semilla : float
        Random state para hacer el proceso replicable
    folds_search : int
        Folds para RandomizedSearchCV, la búsqueda de los mejores hiperparámetros
    folds_CV : int
        Folds para CrossValidation después de encontrar los mejores hiperparámetros
    iteraciones : floar
        Valor entre 0 y 1. Porcentaje de iteraciones para RandomizedSearchCV que depende
        de la cardinalidad del espacio de hiperparámetros en el que se busca en cada modelo
    score : str
        Score con el que se evalúa al mejor modelo    
    param_grid: dict
        Espacio de búsqueda de hiperparámetros
    Returns
    -------
    dict
        Diccionario con: objeto del RandomizedSearchCV en el que se encuentra el mejor
        modelo, objeto del mejor modelo, dataframe con resultados del modelo
    """   
    
    
    modelos={}
    tabla_scores=pd.DataFrame(columns=["modelo","score_searched","roc","score_mean_cross_val","score_std_cross_val","accuracy","F1","confusion","ks","thr"])    
      
    xgb = XGBClassifier()
    xgb.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    combinaciones=np.prod(list(map(len, param_grid.values())))
    clf = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, cv=folds_search, scoring=score, 
                       error_score=-1000, n_jobs=-1, verbose=5,
                            n_iter=int(combinaciones*iteraciones)+1, random_state=semilla)
    clf.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])  
    xgb_ajustado = XGBClassifier(**clf.best_params_)
    xgb_ajustado.fit(X_entrenamiento, y_entrenamiento[y_entrenamiento.columns[0]])
    ls_medias = cross_val_score(estimator=xgb_ajustado, X=X_entrenamiento, y = y_entrenamiento[y_entrenamiento.columns[0]], cv = folds_CV, n_jobs=-1, scoring=score)
    
    fpr, tpr, thresholds = roc_curve(y_entrenamiento,xgb_ajustado.predict_proba(X_entrenamiento)[:,1])
    Gmeans = tpr * (1-fpr)
    thr = np.argmax(Gmeans)
    thr = thresholds[thr]
    
    tabla_scores.loc[len(tabla_scores)] =["xgb_classifier",
                                          clf.best_score_,
                                          roc_auc_score(y_score=xgb_ajustado.predict_proba(X_prueba)[:,1],y_true=y_prueba) if score=="roc_auc" else 0,            
                                          np.mean(ls_medias), 
                                          np.std(ls_medias),
                                          accuracy_score(y_pred=xgb_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          f1_score(y_pred=xgb_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba,average="binary" if score=="roc_auc" else 'weighted'),
                                          confusion_matrix(y_pred=xgb_ajustado.predict_proba(X_prueba)[:,1]>thr,y_true=y_prueba),
                                          ks_2samp(xgb_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==0],xgb_ajustado.predict_proba(X_prueba)[:,1][y_prueba[y_prueba.columns[0]]==1])[0],
                                          thr
                                         ]
    modelos["xgb_classifier"]={"modelo":clf,
                    "parametros":clf.best_estimator_}
    
    tabla_scores=tabla_scores.sort_values(by="score_searched",ascending=False)
    modelos["tabla_scores"]=tabla_scores
    
    
    return(modelos)