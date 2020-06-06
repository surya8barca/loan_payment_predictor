from flask import Flask,jsonify,request
import joblib
import pandas as pd
from flask.templating import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('loan.html')

@app.route("/predict/",methods=['GET'])
def predict():
    from_form=request.args
    print(from_form)
    model=None
    data_for_prediction=None
    model_type=''
    result=''
    rate=float(from_form['int_rate'])/100
    
    if(from_form['model']=='dt'):
        model_type='Dicision Tree'
        model=joblib.load('Decision_Tree_model.sav')
    else:
        model_type='Random Forest Classifier'
        model=joblib.load('Random_Forest_model.sav')
    if(from_form['purpose']=='1'):
        data_for_prediction=[[int(from_form['policy']),rate,float(from_form['installment']),float(from_form['natural_log']),float(from_form['dti']),int(from_form['fico']),int(from_form['cline']),int(from_form['rbal']),float(from_form['rutil']),int(from_form['inq6m']),int(from_form['dead2y']),int(from_form['pubrec']),1,0,0,0,0,0]]                                                                                                    
    elif(from_form['purpose']=='2'):
        data_for_prediction=[[int(from_form['policy']),rate,float(from_form['installment']),float(from_form['natural_log']),float(from_form['dti']),int(from_form['fico']),int(from_form['cline']),int(from_form['rbal']),float(from_form['rutil']),int(from_form['inq6m']),int(from_form['dead2y']),int(from_form['pubrec']),0,1,0,0,0,0]]                                                                                                    
    elif(from_form['purpose']=='3'):
        data_for_prediction=[[int(from_form['policy']),rate,float(from_form['installment']),float(from_form['natural_log']),float(from_form['dti']),int(from_form['fico']),int(from_form['cline']),int(from_form['rbal']),float(from_form['rutil']),int(from_form['inq6m']),int(from_form['dead2y']),int(from_form['pubrec']),0,0,1,0,0,0]]                                                                                                    
    elif(from_form['purpose']=='4'):
        data_for_prediction=[[int(from_form['policy']),rate,float(from_form['installment']),float(from_form['natural_log']),float(from_form['dti']),int(from_form['fico']),int(from_form['cline']),int(from_form['rbal']),float(from_form['rutil']),int(from_form['inq6m']),int(from_form['dead2y']),int(from_form['pubrec']),0,0,0,1,0,0]]                                                                                                    
    elif(from_form['purpose']=='5'):
        data_for_prediction=[[int(from_form['policy']),rate,float(from_form['installment']),float(from_form['natural_log']),float(from_form['dti']),int(from_form['fico']),int(from_form['cline']),int(from_form['rbal']),float(from_form['rutil']),int(from_form['inq6m']),int(from_form['dead2y']),int(from_form['pubrec']),0,0,0,0,1,0]]                                                                                                    
    elif(from_form['purpose']=='6'):
        data_for_prediction=[[int(from_form['policy']),rate,float(from_form['installment']),float(from_form['natural_log']),float(from_form['dti']),int(from_form['fico']),int(from_form['cline']),int(from_form['rbal']),float(from_form['rutil']),int(from_form['inq6m']),int(from_form['dead2y']),int(from_form['pubrec']),0,0,0,0,0,1]]                                                                                                    
    
    prediction=model.predict(data_for_prediction)
    if(prediction[0]==0):
        result='Loan Not Fully Paid'
    else:
        result='Loan Fully Paid'
    return jsonify({'model': model_type,'outcome': result}) 


if __name__ == '__main__':
    app.run()