from distutils.log import debug # va aidé à 
from flask import Flask,request,jsonify, render_template,redirect,url_for
import sklearn
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__) # pour preparer l'environnement
models=pickle.load(open('ModelDiabete.pkl','rb'))# pour reconnaitre les elements du target
dict_class_lesion={
	0:"Negatif",
	1:"positif"	
}
@app.route('/') # pour avoir la route
def home():
	return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
	models=pickle.load(open('ModelDiabete.pkl','rb')) #chergement du model
	int_futures=[float(i)for i in request.form.values()] #converssion des donnees
	dernier_futures=[np.array(int_futures)] #donner la forme aux donnees de mm maniere pytho
	dernier_futures=np.array([dernier_futures]).reshape(1,8) # on prend tout 
	predire=models.predict(dernier_futures)
	pred_class=predire.argmax(axis=-1) #toutes les colonnes moins le target
	prediction=dict_class_lesion[predire[0]] # 0 pour unitialiser lors de la prediction
	resultat=str(prediction)
	return render_template('index.html',prediction_text_="votre type est:{}".format(resultat))
if __name__=='__main__':
	app.run(debug=True) # debug permet de detecter l'erreur


