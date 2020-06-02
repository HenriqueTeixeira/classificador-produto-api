# coding: utf-8
import os
import pandas as pd
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from sklearn.externals import joblib

#nltk.download('stopwords')
#nltk.download('rslp')
#nltk.download('punkt')

#carregando o modelo em memoria
modelo = joblib.load('modelo/modelo_produto_27.pkl')

def predicao(url):
    count = 0
    pipe = modelo
    words = pre_processamento(url)
    resp = pipe.predict(words)
    print "O Código do produto escolhido foi: %s \n" % resp[0]
    resp = zip(pipe.classes_, pipe.predict_proba(words)[0])
    resp.sort(key=lambda tup: tup[1], reverse=True)

    return resp

def to_unicode(data):
    try:
        data = data.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        try:
            data = data.decode('iso-8859-1')
        except (UnicodeDecodeError, UnicodeEncodeError):
            try:
                data = data.decode('latin-1')
            except (UnicodeDecodeError, UnicodeEncodeError):
                data = data
    return data

def chr_remove(old, to_remove):
    new_string = old
    for x in to_remove:
        new_string = new_string.replace(x, ' ')
    return new_string

def pre_processamento(desc):
    stops = set(stopwords.words("portuguese"))
    text = desc
    words = text.lower()
    words = chr_remove(words, "!@#$%^&*()[]{};,./<>?\|`~-=_+")
    words = nltk.tokenize.word_tokenize(to_unicode(words))

    return words

def worker(ITEM_DESC, TARGET, line):
    words = pre_processamento(ITEM_DESC)
    print "{:6d} words in: \t {:.70}".format(len(words), ITEM_DESC)
    line.append((ITEM_DESC, TARGET, words))

# Instancia o Flask
app = Flask(__name__)

@app.route('/definition', methods=['POST'])
def definition():
    definition = "API REST que retorna um post contendo a classificação dos produtos de peças e acessórios com base na" \
                 "descrição dos itens da NF-e."


@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    # Coletando os Dados
    if test_json:
        if isinstance( test_json, dict ): # unique value
            df_row = pd.DataFrame( test_json, index=[0])
        else:
            df_row = pd.DataFrame( test_json, columns=test_json[0].keys() )

    #Predição da requisição
    resultado = []
    for n in range(0,len(df_row)):
        textoPredicao = df_row.iloc[n]['descricao']
        pred = predicao(textoPredicao)

        codProdutos = [pruduto for pruduto, probabilidade in pred]
        proba = [probabilidade for pruduto, probabilidade in pred]

    # -----------------------------------------------------------------------------------
        resultado.append( {'codProduto': str(codProdutos[0]),
                           'descProduto':  textoPredicao,
                           'acuracia' : round(proba[0]*100,2),
                           'outros': [{
                                       'produto': str(codProdutos[1]),
                                       'acuracia': round(proba[1]*100,2)
                                       },
                                       {
                                        'produto': str(codProdutos[2]),
                                        'acuracia': round(proba[2]* 100,2)
                                       }
                                      ]
                           })

    return jsonify(resultado)

if __name__ == '__main__':
    #start Flask
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
