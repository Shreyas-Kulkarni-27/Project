import numpy as np
import scipy as sp
import pandas as pd 
import os
import matplotlib.pyplot as plt
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load (open ('model.pkl','rb'))

@app.route ('/')
def home():
    return render_template ('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    file = request.files['file']
    file.save(os.path.join('C:/Users/Shreyas Kulkarni/OneDrive/Desktop/Project/flask demo', file.filename))
    df = pd.read_csv(file.filename)
    
    plt.hist(df['Age'], bins=10)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Distribution of Age')
    # plt.show()
    plt.savefig('demo12wu3_plot.png')
    
    
    df['Attrition'].replace('Yes','1',inplace=True)
    df['Attrition'].replace('No','0',inplace=True) 
    df.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber'],axis=1,inplace=True)

    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60], labels=['18-30','31-40','41-50','51-60'])
    age_group_counts = df['AgeGroup'].value_counts()

    age_group_attrition_counts = df[df['Attrition'] == '1'].groupby('AgeGroup')['Attrition'].count()
    age_group_attrition_counts

    age_group_counts = df['AgeGroup'].value_counts()
    age_group_attrition_counts = df[df['Attrition'] == '1'].groupby('AgeGroup')['Attrition'].count()

    age_group_counts = age_group_counts.reindex(age_group_attrition_counts.index)

    for i,(total, attrition) in enumerate(zip(age_group_counts, age_group_attrition_counts)):
        percentage = attrition / total * 100


    #df.drop(['EmployeeCount', 'Attrition', 'Over18','StandardHours','EmployeeNumber'],axis=1,inplace=True)
    df.duplicated().sum()
    #df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60], labels=['18-30','31-40','41-50','51-60'])

    # df['Attrition'].replace('1','Yes',inplace=True)
    # df['Attrition'].replace('0','No',inplace=True)

    # attrition_dummies = pd.get_dummies(df['Attrition'])

    # df = pd.concat([df, attrition_dummies], axis = 1)

    # df = df.drop(['Attrition', 'No'], axis =1 )
    
    df.drop('AgeGroup',axis=1,inplace=True)
    # df.drop(['Age','JobLevel'], axis = 1)

    print(df.info())

    df['Department'] = df['Department'].apply (lambda x: 1 if x == 'Research & Development' else 0)

    df['BusinessTravel'] = df['BusinessTravel'].apply (lambda x: 1 if x == 'Travel_Rarely' else 0)
    
    df['EducationField'] = df['EducationField'].apply (lambda x: 1 if x == 'Life Sciences' else 0)

    df['Gender'] = df['Gender'].apply (lambda x: 1 if x == 'Male' else 0)

    df['JobRole'] = df['JobRole'].apply (lambda x: 1 if x == 'Sales Executive' else 0)

    df['MaritalStatus'] = df['MaritalStatus'].apply (lambda x: 1 if x == 'Married' else 0)

    df['OverTime'] = df['OverTime'].apply (lambda x: 1 if x == 'No' else 0)

    # from sklearn.preprocessing import LabelEncoder
    # for column in df.columns:
    #     if df[column].dtype == 'object':
    #         # df[column] = LabelEncoder().fit_transform(df[column])
    #         # df['BusinessTravel'] = df['BusinessTravel'].astype('category')
    #         df[column] = LabelEncoder().fit_transform(df[column])
            
    #     else:
    #         continue
    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    # for feature in df.columns:
    #     if df[feature].dtype == "object":
    #          df[feature] = pd.Categorical(df[feature]).codes
    X = df.drop(['Attrition'], axis=1)

    prediction = model.predict(X)
    
    # print(age_group_attrition_counts)
    print(prediction)
    
    count= np.count_nonzero(prediction == 1)
    co = len(prediction)
    for i in range(0, len(prediction)):  
        print(prediction[i]),
    print(count)
    print(co)
    p = count / co * 100
    # unique_counts, count_ones = np.unique(prediction,return_counts=True)
    # tot_p = dict(zip(unique_counts, count_ones))
    # print(tot_p)
    # p = (tot_p[1] / 1470) * 100 
    # p = 10009
    # print(percentage)
    if prediction.any():
        return render_template('show.html',p=p)
    else:
        return render_template ('index.html',prediction_text='Predicted Failed')
  
if __name__ == "__main__":
    app.run (debug=True)
