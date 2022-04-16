import pandas as pd
from flask import Flask,render_template,request
import pickle


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        try:
            Pclass = int(request.form['Pclass'])
            Sex = int(request.form['Sex'])
            Age = float(request.form['Age'])
            SibSp = int(request.form['SibSp'])
            Parch = int(request.form['Parch'])
            Fare = float(request.form['Fare'])
            predict_X = [Pclass,Sex,SibSp,Parch,Fare,Age]
            predict_X_df = pd.DataFrame(predict_X)
            with open("standardScalar.sav",'rb') as f:
                scalar =pickle.load(f)
            predict_X_scaled = scalar.transform([[predict_X_df[0][0],predict_X_df[0][1],predict_X_df[0][2],predict_X_df[0][3],predict_X_df[0][4],predict_X_df[0][5]]])
            with open("modelForPrediction.sav",'rb') as f:
                model = pickle.load(f)
            predict = model.predict(predict_X_scaled)
            print("Prediction value is:",predict)
            if predict[0] == 0:
                result = "Not survived"
            if predict[0] == 1:
                result = "Survived"
            return render_template('results.html',prediction=result)
        except Exception as e:
            print("The exception is:",e)
            return "Something is wrong"
    else:
        return render_template("index1.html")

if __name__ == '__main__':
    app.run(port=5000,debug=True)