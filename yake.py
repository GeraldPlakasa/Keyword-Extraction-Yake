import math
import statistics
# Pickle untuk segmentasi kalimat
import pickle
import numpy as np
# Untuk ambil katrakter spesial
from string import punctuation
# Untuk perhitungan jarak Levenshtein
import Levenshtein as lev
import itertools

import nltk
nltk.download('punkt')

class Yake:

    def __init__(self):
        self.__teks_hasil = []
        self.__special = list(punctuation)
        self.__TF_murni = []
        self.__TF_normalisasi = []
        self.__tf_u_w = []
        self.__tf_a_w = []
        self.__teks_segmentasi_kalimat_token = []
        self.__med_sen_w = []
        self.__stopword = []
        self.__teks_hasil_fix = []
        self.__TF_kw_normalisasi = []
        self.__skw = []

        self.__special.append("`")
        self.__special.append("``")
        self.__special.append("''")

    def keyword(self, teks_dataset, n = 100):

        self.__teks_dataset = teks_dataset
        self.__preProcessing()
        self.__setFrequency()
        self.__featureExtraction()
        self.__individualTermWeighting()
        self.__candidateKeywordListGeneration()
        self.__levenshteinDistance()
        hasil_dict = self.getAllKeyword()
        hasil_dict_sort = dict(sorted(hasil_dict.items(), key=lambda x: x[1]))
        if n > len(hasil_dict_sort):
            return hasil_dict_sort
        else:
            return dict(itertools.islice(hasil_dict_sort.items(), n))

    def __preProcessing(self):
        # Tokenisasi
        self.__teks_tokenisasi = nltk.word_tokenize(self.__teks_dataset)

        # Hapus karakter Spesial
        for teks in self.__teks_tokenisasi:
            teks = teks.lower()
            if teks not in self.__teks_hasil:
                if teks not in self.__special:
                    self.__teks_hasil.append(teks)

    def __featureExtraction(self):
        # Word Casing
        u_w, a_w = self.__setUwAw()
        self.__setTFUwTFAw(u_w, a_w)
        self.__w_case = [max(self.__tf_u_w[i],self.__tf_a_w[i])/math.log(self.__TF_normalisasi[i],2) for i in range(len(u_w))]

        # Word Position
        self.__segmentasiKalimat()
        self.__medianSenW()
        self.__w_position = [math.log(math.log(2 + nilai,2),2) for nilai in self.__med_sen_w]

        # Word Frequency
        self.__w_frequency = [nilai/(statistics.mean(self.__TF_normalisasi) + 1*statistics.stdev(self.__TF_normalisasi)) for nilai in self.__TF_normalisasi]

        # Word Relatedness to Context
        wl, wr, pl, pr = self.__setWlWrPlPr()
        self.__w_rel = [(0.5 + ((wl[i]*self.__TF_normalisasi[i]/max(self.__TF_normalisasi)) + pl[i])) + (0.5 + ((wr[i]*self.__TF_normalisasi[i]/max(self.__TF_normalisasi)) + pr[i])) for i in range(len(wl))]

        # Word DifSentence
        sf = self.__setSf()
        self.__w_dif = [nilai/len(self.__teks_segmentasi_kalimat) for nilai in sf]

    def __individualTermWeighting(self):
        self.__sw = [self.__w_rel[i]*self.__w_position[i]/(self.__w_case[i] + (self.__w_frequency[i]/self.__w_rel[i]) + (self.__w_dif[i]/self.__w_rel[i])) for i in range(len(self.__teks_hasil))]

    def __candidateKeywordListGeneration(self):
        self.__setStopword()
        self.__setTextFix()
        self.__deleteDirtyTerm()
        self.__setKWFrequency()
        self.__setSKW()

    def __levenshteinDistance(self):
        idxs = []
        for i,txt_pertama in enumerate(self.__teks_hasil_fix):
          for j,txt_kedua in enumerate(self.__teks_hasil_fix):
            jarak = lev.distance(txt_pertama, txt_kedua)
            if jarak < 2 and not i == j:
              if self.__skw[i] < self.__skw[j]:
                idxs.append(j)
              else:
                idxs.append(i)
        idxs = list(set(idxs))
        idxs.sort()

        self.__keyword = self.__teks_hasil_fix.copy()
        self.__keyword_skw = self.__skw.copy()
        for i in range(len(idxs),0,-1):
            self.__keyword.pop(i)
            self.__keyword_skw.pop(i)

    def getTokenisasi(self):
        return self.__teks_tokenisasi
    def getFrequency(self):
        frekuensi = dict(zip(self.__teks_hasil, self.__TF_murni))
        return frekuensi
    def getWCase(self):
        w_case = dict(zip(self.__teks_hasil, self.__w_case))
        return w_case
    def getWPosition(self):
        w_position = dict(zip(self.__teks_hasil, self.__w_position))
        return w_position
    def getWFrequency(self):
        w_frequency = dict(zip(self.__teks_hasil, self.__w_frequency))
        return w_frequency
    def getWRel(self):
        w_rel = dict(zip(self.__teks_hasil, self.__w_rel))
        return w_rel
    def getWDif(self):
        w_dif = dict(zip(self.__teks_hasil, self.__w_dif))
        return w_dif
    def getSw(self):
        sw = dict(zip(self.__teks_hasil, self.__sw))
        return sw
    def getSkw(self):
        skw = dict(zip(self.__teks_hasil_fix, self.__skw))
        return skw
    def getAllKeyword(self):
        keyword = dict(zip(self.__keyword, self.__keyword_skw))
        return keyword

    def __setFrequency(self):
        # TF diskrit
        self.__teks_tokenisasi_temp = [teks.lower() for teks in self.__teks_tokenisasi]
        for teks in self.__teks_hasil:
            self.__TF_murni.append(self.__teks_tokenisasi_temp.count(teks))

        self.__TF_normalisasi = []
        # TF Normalisasi
        for nilai in self.__TF_murni:
            self.__TF_normalisasi.append(nilai/sum(self.__TF_murni))

    def __setUwAw(self):
        u_w = []
        a_w =  []
        for term in self.__teks_hasil:
            # Ambil seluruh index lokasi kata
            idxs_teks = [i for i,teks in enumerate(self.__teks_tokenisasi) if teks.lower() == term]
            u_w_temp = 0
            a_w_temp = 0
            for idx in idxs_teks:
                # Jika term bukan di awal kalimat dan term diawali kapital dan bukan dengan huruf kapital semua
                if idx != 0 and not self.__teks_tokenisasi[idx-1] == '.' and self.__teks_tokenisasi[idx][0].isupper() and not self.__teks_tokenisasi[idx].isupper():
                    u_w_temp += 1
                if self.__teks_tokenisasi[idx].isupper():
                    a_w_temp += 1

            u_w.append(u_w_temp)
            a_w.append(a_w_temp)

        return (u_w, a_w)

    def __setTFUwTFAw(self, u_w, a_w):
        self.__tf_u_w = []
        for nilai in u_w:
            if sum(u_w) == 0:
                self.__tf_u_w.append(0)
            else:
                self.__tf_u_w.append(nilai/sum(u_w))

        self.__tf_a_w = []
        for nilai in a_w:
            if sum(a_w) == 0:
                self.__tf_a_w.append(0)
            else:
                self.__tf_a_w.append(nilai/sum(a_w))

    def __segmentasiKalimat(self):
        tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        tokenizer.train(self.__teks_dataset)
        out = open("indonesian.pickle", "wb")
        pickle.dump(tokenizer, out)
        out.close()
        seg_kalimat = nltk.data.load('indonesian.pickle')
        self.__teks_segmentasi_kalimat = seg_kalimat.tokenize(self.__teks_dataset)

        for i in range(len(self.__teks_segmentasi_kalimat)):
            temp = nltk.word_tokenize(self.__teks_segmentasi_kalimat[i])
            temp2 = [teks.lower() for teks in temp]
            self.__teks_segmentasi_kalimat_token.append(temp2)

    def __medianSenW(self):
        sen_w_last = [1]
        for i,term in enumerate(self.__teks_hasil):
            sen_w = []
            for j, kalimat in enumerate(self.__teks_segmentasi_kalimat_token):
                if term in kalimat:
                    sen_w.append(j+1)

            # Jika sen_w kosong, ambil nilai sen_w sebelumnya
            if not sen_w:
                self.__med_sen_w.append(statistics.median(sen_w_last))
                sen_w_last = sen_w_last.copy()
            else:
                self.__med_sen_w.append(statistics.median(sen_w))
                sen_w_last = sen_w.copy()

    def __setWlWrPlPr(self):
        wl = []
        pl = []
        wr = []
        pr = []

        for i,teks in enumerate(self.__teks_hasil):
            if self.__TF_murni[i] > 1:
                idx = [i for i,x in enumerate(self.__teks_tokenisasi) if x==teks]
                cek_teks_wl = []
                sama_wl = 0
                cek_teks_wr = []
                sama_wr = 0

                for index in idx:
                    if index == 0 or index == len(self.__teks_tokenisasi)-1:
                        pass
                    else:
                        # Cek kata di kiri
                        if self.__teks_tokenisasi[index-1] not in cek_teks_wl:
                            cek_teks_wl.append(self.__teks_tokenisasi[index-1])
                        else:
                            sama_wl += 1
                        # Cek kata di kanan
                        if self.__teks_tokenisasi[index+1] not in cek_teks_wr:
                            cek_teks_wr.append(self.__teks_tokenisasi[index+1])
                        else:
                            sama_wr += 1

                # Masukan WL
                if sama_wl == 0:
                    wl.append(0)
                else:
                    wl.append(len(cek_teks_wl)/sama_wl)

                # Masukan PL
                pl.append(len(cek_teks_wl)/max(self.__TF_normalisasi))

                # Masukan WR
                if sama_wr == 0:
                    wr.append(0)
                else:
                    wr.append(len(cek_teks_wr)/sama_wr)

                # Masukan PR
                pr.append(len(cek_teks_wr)/max(self.__TF_normalisasi))
            else:
                wl.append(0)
                wr.append(0)
                pl.append(0)
                pr.append(0)

        return wl,wr,pl,pr

    def __setSf(self):
        sf = []
        for teks in self.__teks_hasil:
            jumlah = 0
            for kalimat in self.__teks_segmentasi_kalimat_token:
                if teks in kalimat:
                    jumlah += 1
            sf.append(jumlah)
        return sf

    def __setStopword(self):
        self.__stopword = []
        f = open("stopword.txt", "r")
        for x in f:
            if x[-1:] == '\n':
                self.__stopword.append(x[:-1])
            else:
                self.__stopword.append(x)

    def __setTextFix(self):
        
        teks_hasil_fix_temp = self.__teks_hasil.copy()
        for i in range(len(self.__teks_tokenisasi_temp)-2):
            if not (self.__teks_tokenisasi_temp[i] in self.__stopword and self.__teks_tokenisasi_temp[i+2] in self.__stopword):
                if not (self.__teks_tokenisasi_temp[i] in self.__special or self.__teks_tokenisasi_temp[i+1] in self.__special or self.__teks_tokenisasi_temp[i+2] in self.__special):
                    if self.__teks_tokenisasi_temp[i] in self.__stopword:
                        teks_hasil_fix_temp.append(self.__teks_tokenisasi_temp[i+1]+" "+self.__teks_tokenisasi_temp[i+2])
                    elif self.__teks_tokenisasi_temp[i+2] in self.__stopword:
                        teks_hasil_fix_temp.append(self.__teks_tokenisasi_temp[i]+" "+self.__teks_tokenisasi_temp[i+1])
                    else:
                        teks_hasil_fix_temp.append(self.__teks_tokenisasi_temp[i]+" "+self.__teks_tokenisasi_temp[i+1]+" "+self.__teks_tokenisasi_temp[i+2])

        # Hapus Duplicate
        for data in teks_hasil_fix_temp:

            if data not in self.__teks_hasil_fix:
                self.__teks_hasil_fix.append(data)

    def __deleteDirtyTerm(self):
        idxs = []
        for i,teks in enumerate(self.__teks_hasil_fix):
            if len(teks.split()) == 1:
                if teks in self.__stopword:
                    idxs.append(i)
                if teks.isdigit():
                    idxs.append(i)

        idxs = list(set(idxs))
        idxs.sort()

        for i in range(len(idxs)-1,0,-1):
            self.__teks_hasil_fix.pop(idxs[i])
            self.__TF_murni.pop(idxs[i])

    def __setKWFrequency(self):
        # set Frequency KW
        self.__TF_kw_murni = self.__TF_murni.copy()
        for i,teks in enumerate(self.__teks_hasil_fix):
            if i > len(self.__TF_murni)-1:
                if self.__teks_dataset.lower().count(teks) <= 0:
                    self.__TF_kw_murni.append(1)
                else:
                    self.__TF_kw_murni.append(self.__teks_dataset.lower().count(teks))

        for nilai in self.__TF_kw_murni:
            self.__TF_kw_normalisasi.append(nilai/sum(self.__TF_kw_murni))

    def __setSKW(self):
        for i,teks_1 in enumerate(self.__teks_hasil_fix):
          teks_temp = teks_1.split()
          sw_temp = []
          for teks_2 in teks_temp:
            idx = self.__teks_hasil.index(teks_2)
            sw_temp.append(self.__sw[idx])
          self.__skw.append(np.prod(sw_temp)/self.__TF_kw_normalisasi[i]*(1+sum(sw_temp)))