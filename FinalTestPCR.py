import pickle
import pandas as pd
import numpy as np

#loading features,imputer and the model
with open('features.pkl', 'rb') as file:
    features_list = pickle.load(file)
    
with open('imputer.pkl', 'rb') as file:
    impute = pickle.load(file)
    
with open('best_lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('minmax.pkl', 'rb') as file:
    scaler = pickle.load(file)

#upload the test file here 
url2 = 'https://raw.githubusercontent.com/ArvinCorotana/ML/main/TestDatasetExample1.xls'
test = pd.read_excel(url2, sheet_name='Sheet1')


#data-preprocessing
test.replace(999,np.nan,inplace=True)   
output = test[['ID']]
test.drop('ID',axis=1,inplace=True)
X_new = impute.transform(test)
X_new = pd.DataFrame(X_new, columns=test.columns, index=test.index)
X_mri = X_new.iloc[:,10:]
X_mri_scaled =scaler.transform(X_mri)
X_mri = pd.DataFrame(X_mri_scaled, columns=X_mri.columns)
X_new = pd.concat((X_new.iloc[:,:10],X_mri),axis=1)
X_test = X_new[features_list]

#model predictions
predictions = model.predict(X_test)
output['PCR'] = predictions
output.to_csv('Test.csv',index=False)