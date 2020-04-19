import numpy as np
from flask import Flask, request, jsonify, render_template
import pymysql
import pandas as pd
import pickle
import re
import string
import nltk
from CobaVectorizer import MeanEmbeddingVectorizer
import gensim
from gensim.models import FastText

app = Flask(__name__, static_folder='static', template_folder='templates') #Initialize the flask App
model = pickle.load(open('model_rf_byu200_02TS_Normal.pkl', 'rb'))
loc = "FastTextModels/saved_model_gensim200SG_BYU.bin"
model_ft = FastText.load(loc)
connection = pymysql.connect(host='localhost', user='root', password='', database='sentimen')
count = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    
    isi = request.form["Comment"]
    temp_char = ''

    temp_isi = isi.lower()
    temp_isi = re.sub(r"\d", "", temp_isi)
    temp_isi = re.sub(r'(?<=[,.])(?=[^\s])', r' ', temp_isi)
    temp_isi = temp_isi.translate(str.maketrans('', '', string.punctuation))
    temp_isi = nltk.tokenize.word_tokenize(temp_isi)
    temp_isi = [temp_isi]


    mean_vec_tr = MeanEmbeddingVectorizer(model_ft)
    hasil_vector = mean_vec_tr.transform(temp_isi)
    prediksi = model.predict(hasil_vector)
    if(prediksi==1):
        temp_char = 'Positif'
    elif(prediksi==0):
        temp_char = 'Netral'
    else:
        temp_char = 'Negatif'
    
    return render_template('index.html', comment_content="COMMENT: {}".format(isi), prediction_text="SENTIMEN: {}".format(temp_char))

@app.route('/upload_file')
def upload_file():
    return render_template('upload.html')

@app.route("/showData", methods=['GET', 'POST'])
def showData():
    file = request.files['inputFile']
    dfweb = pd.read_excel(file)
    cursor = connection.cursor()

    sql = "DROP TABLE IF EXISTS `data`"
    cursor.execute(sql)
    sql = "CREATE TABLE data (COMMENT TEXT);"
    cursor.execute(sql)

    cols = "``".join([str(i) for i in dfweb.columns.tolist()])

    for i,row in dfweb.iterrows():
        sql = "INSERT INTO `data` (`" +cols + "`) VALUES (" + "%s"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))
        connection.commit()
    
    # connection.close()

    return render_template(
        # "showData.html",
        "upload.html",
        total=len(dfweb),
        tables=[dfweb.to_html(classes="table table-striped", header="true", index=False)],
        titles=dfweb.columns.values,
        shape=count
    )

@app.route("/predicted", methods=['GET', 'POST'])
def predicted():
    cursor2 = connection.cursor()
    sql = "SELECT * from `data`"
    cursor2.execute(sql)
    result = cursor2.fetchall()
    texts = []
    pos = 0
    net = 0

    for i in result:
        texts.append(i)

    #pre-process
    dfprint = pd.DataFrame()
    dfprint['COMMENT'] = texts

    text_baru = dfprint['COMMENT'].astype(str)    
    text_baru = text_baru.apply(lambda x: x.lower()) #Lower Case
    text_baru = text_baru.apply(lambda x: re.sub(r"\d", "", x)) #Remove Number    
    text_baru = text_baru.apply(lambda x: re.sub(r'(?<=[,.])(?=[^\s])', r' ', x)) #Before Punctuation
    text_baru = text_baru.apply(lambda x: x.translate(str.maketrans('','',string.punctuation))) #punctuation    
    text_baru = text_baru.apply(lambda x: nltk.tokenize.word_tokenize(x)) #tokenizing

    dfprint['tokenized'] = text_baru
    dfprint = dfprint.dropna()

    #prediction process
    mean_vec_tr = MeanEmbeddingVectorizer(model_ft)
    hasil_vector = mean_vec_tr.transform(dfprint['tokenized'])
    predictions = model.predict(hasil_vector)

    dfprint['sentimen'] = predictions
    dfprint['sentimen'] = dfprint['sentimen'].replace([1, 0, -1], ['Positif', 'Netral', 'Negatif'])
    dfpred = dfprint[['COMMENT', 'sentimen']]
    
    # connection.close()

    totalData = len(predictions)
    for j in predictions:
        if j == 1:
            pos = pos + 1
        elif j == 0:
            net += 1
    neg = totalData - pos - net

    posPercentage = (pos/totalData)*100
    negPercentage = (neg/totalData)*100
    netPrecentage = (net/totalData)*100

    donePredict = True

    return render_template(
        # "predicted.html",
        "upload.html",
        posP = posPercentage,
        negP = negPercentage,
        netP = netPrecentage,
        tables2=[dfpred.to_html(classes="table table-striped", header="true", index=False)],
        titles=dfpred.columns.values
    )

    
if __name__ == "__main__":
    app.run(debug=True)