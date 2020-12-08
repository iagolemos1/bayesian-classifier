import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

class BayesianClassifier(object):
    
    def pre_processor(data, train_set_p):
        classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)  
        data = data.to_numpy()
        nrow,ncol = data.shape
        y = data[:,-1]
        X = data[:,0:ncol-1]
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = train_set_p)
        return(x_train, x_test, y_train, y_test, classes, data, scaler)

    def fit_model(pre_processed_data):
        x_train, x_test, y_train, y_test, classes, data, scaler = pre_processed_data
        P = pd.DataFrame(data=np.zeros((x_test.shape[0], len(classes))), columns = classes) 
        Pc = np.zeros(len(classes))
        for i in np.arange(0, len(classes)):
            elements = tuple(np.where(y_train == classes[i]))
            Pc[i] = len(elements)/len(y_train)
            Z = x_train[elements,:][0]
            m = np.mean(Z, axis = 0)
            cv = np.cov(np.transpose(Z))
            for j in np.arange(0,x_test.shape[0]):
                x = x_test[j,:]
                pj = multivariate_normal.pdf(x, mean=m, cov=cv, allow_singular=True)
                P[classes[i]][j] = pj*Pc[i]
        y_pred = []
        for i in np.arange(0, x_test.shape[0]):
            c = np.argmax(np.array(P.iloc[[i]]))
            y_pred.append(classes[c])
        y_pred = np.array(y_pred, dtype=str)
        score = accuracy_score(y_pred, y_test)
        print('Accuracy:', score)
        return(Pc, Z, m, cv, classes, scaler)
        
    def predict(fitted_model, val):
        Pc, Z, m, cv, classes, scaler = fitted_model
        x = scaler.transform(val)
        pj = multivariate_normal.pdf(x, mean=m, cov=cv, allow_singular = True)
        P = []
        for j in range(0, len(classes)):
            P.append(pj*Pc[j])
        c = np.argmax(np.array(P))
        y_pred = classes[c]
        return(y_pred)
        
    
        



data_= pd.read_csv('Vehicle.csv', header=(0))
data_ = data_.dropna(axis='rows')
ppd = BayesianClassifier.pre_processor(data_, 0.8)

fitted_model = BayesianClassifier.fit_model(ppd)

data_pred = data_.to_numpy()[0,:-1].reshape(1,-1)

pred = BayesianClassifier.predict(fitted_model, data_pred)