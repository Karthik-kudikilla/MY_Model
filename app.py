from flask import Flask,request,render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=["POST","GET"])
def predict():
    features=[x for x in request.form.values()]
    print(features)
    df = pd.read_csv('Admission_Predict_Ver1.1 (1).csv')
    df.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)
    df.drop(labels='Serial No.', axis=1, inplace=True)

    targets = df['Chance of Admit']
    X= df.drop(columns={'Chance of Admit'})
    scaler = StandardScaler()


    final=[np.array(features)]
    #x=[[320,120,5,4,4,9,1]]
    sst=StandardScaler().fit(X)
    output=model.predict(sst.transform(final))
    print(output)
    if output>=0.5:
     return render_template('index.html', pred=f'ELIGIBLE, {output}')
    else:
        return render_template('index.html', pred=f'NOT ELIGIBLE, {output}')

if __name__=='__main__':
    app.run(debug=True)