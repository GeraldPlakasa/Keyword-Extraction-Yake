from flask import Flask, render_template, request
import yake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

yake_obj = yake.Yake()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/demo")
def demo():
    return render_template('demo.html')

@app.route("/hasil", methods=['GET', 'POST'])
def hasil():

    judul=request.form['judul']
    abstrak=request.form['abstrak']
    n=request.form['myRange']

    yake_obj = yake.Yake()

    teks_dataset = judul.title() + ". "
    for teks in abstrak.split('\n'):
      teks_dataset = teks_dataset + teks + " "

    hasil = yake_obj.keyword(teks_dataset, int(n))

    keyphrase = list(hasil.keys())

    all_idx = [i for i in range(len(abstrak)) if abstrak.lower().startswith(keyphrase[0], i)]
    for idx in reversed(all_idx):
        abstrak = abstrak[:idx]+ '<span style="background-color:#47b2e4">' + abstrak[idx:idx+len(keyphrase[0])] + '</span>' + abstrak[idx+len(keyphrase[0]):]

    all_idx = [i for i in range(len(judul)) if judul.lower().startswith(keyphrase[0], i)]
    for idx in reversed(all_idx):
        judul = judul[:idx]+ '<span style="background-color:#47b2e4">' + judul[idx:idx+len(keyphrase[0])] + '</span>' + judul[idx+len(keyphrase[0]):]

    return render_template('hasil.html',bobot=hasil, keywords=keyphrase, abstrak=abstrak, judul=judul, enumerate=enumerate, round=round)

@app.route("/evaluasi")
def evaluasi():
    
    file = open('static/data/judul.txt', encoding='utf8')
    Lines_judul = file.readlines()
    daftar_judul = []
    for judul in Lines_judul:
        daftar_judul.append(judul.strip().title())

    return render_template('evaluasi.html', daftar_judul=daftar_judul, enumerate=enumerate)

@app.route('/detail/<id>')
def detail(id):
    
    file3 = open('static/data/judul.txt', encoding='utf8')
    Lines_judul = file3.readlines()
    daftar_judul = []
    for judul in Lines_judul:
        daftar_judul.append(judul.strip().title())
    
    file2 = open('static/data/data'+id+'.txt', encoding='utf8')
    Lines = file2.readlines()
    abstrak_temp = ""
    for line in Lines:
        abstrak_temp += line.strip() + " "
    judul_temp = daftar_judul[int(id)]

    file1 = open('static/data/golden'+id+'.txt', encoding='utf8')
    Lines_golden = file1.readlines()
    golden = ""
    for line in Lines_golden:
        golden += line.strip() + " "

    teks_dataset = judul_temp +". " + abstrak_temp.strip()

    yake_obj = yake.Yake()

    hasil = yake_obj.keyword(teks_dataset, 500)

    keyphrase = list(hasil.keys())
    abstrak = abstrak_temp
    judul = judul_temp

    kata_kunci = Lines_golden[0].split(",")
    # proses evaluasi
    kata_kunci_lower = [kata.lower().strip() for kata in kata_kunci]
    panjang_kata_kunci = len(kata_kunci_lower)
    all_kata = kata_kunci_lower + keyphrase

    vectorizer = CountVectorizer().fit_transform(all_kata)
    # menjadikan vectorizernya array
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)

    masuk_threshold_temp = []
    for i in range(panjang_kata_kunci):
      idx = np.where(csim[i]>0.8)
      for j in idx[0]:
        if j >= panjang_kata_kunci:
          masuk_threshold_temp.append(all_kata[j])

    masuk_threshold = []
    for kata in masuk_threshold_temp:
      if kata not in masuk_threshold:
        masuk_threshold.append(kata.strip())
    print(kata_kunci)
    print(masuk_threshold)
    TP = 0
    FN = 0
    kata_sama = []
    for kata in kata_kunci:
      if kata.lower().strip() in masuk_threshold:
        TP += 1
        kata_sama.append(kata)
      else:
        FN += 1
    FP = len(masuk_threshold) - len(kata_sama)
    TN = len(keyphrase) - len(masuk_threshold)
    print(kata_sama)

    try:
      precision = TP/(TP+FP)
    except:
      precision = 0
    try:
      recall = TP/(TP+FN)
    except:
      recall = 0
    try:
      f_score = 2*precision*recall/(precision+recall)
    except:
      f_score = 0

    accuracy = (TP + TN)/(TP+TN+FP+FN)
    
    if len(masuk_threshold)>0:
        # proses highligh kata
        all_idx = [i for i in range(len(abstrak)) if abstrak.lower().startswith(masuk_threshold[0], i)]
        for idx in reversed(all_idx):
            abstrak = abstrak[:idx]+ '<span style="background-color:#47b2e4">' + abstrak[idx:idx+len(masuk_threshold[0])] + '</span>' + abstrak[idx+len(masuk_threshold[0]):]
    
        all_idx = [i for i in range(len(judul)) if judul.lower().startswith(masuk_threshold[0], i)]
        for idx in reversed(all_idx):
            judul = judul[:idx]+ '<span style="background-color:#47b2e4">' + judul[idx:idx+len(masuk_threshold[0])] + '</span>' + judul[idx+len(masuk_threshold[0]):]

    return render_template('detail.html', judul=judul, golden=golden, abstrak=abstrak,
                           enumerate=enumerate, keywords=masuk_threshold, TP=TP, FN=FN, FP=FP, TN=TN,
                           precision=precision, recall=recall, f_score=f_score, accuracy=accuracy,round=round, id=int(id))