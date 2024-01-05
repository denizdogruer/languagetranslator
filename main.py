import tkinter as tk
from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("languageset.csv")

pencere = tk.Tk()
pencere.title("Dil Algılama ve Çevirme Uygulaması")
pencere.geometry('600x400')
pencere.configure(bg='#000000')  # Arka plan rengi

metin = "Dil Algılama ve Çevirme Uygulamasına Hoş Geldiniz!\n"\
        "Aşağıdaki boş alana herhangi bir dilde metin giriniz.\n"
etiket = tk.Label(pencere, text=metin, fg='#FFFFFF', bg='#000000', font=('Helvetica', 14, 'normal'), justify='center')
etiket.pack(pady=10)

# Metin giriş alanını genişletme
kelimeGir = tk.Entry(pencere, width=50, bg='#333333', fg='#FFFFFF')  # Giriş alanının arka plan rengi ve yazı rengi
kelimeGir.pack(pady=10)

def diliAlgila(data):
    user = kelimeGir.get()
    x = np.array(data["Text"])
    y = np.array(data["language"])
    cv = CountVectorizer()
    X = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    data = cv.transform([user]).toarray()
    output = model.predict(data)
    sonuc["text"] = output[0]

def dilCevir(data):
    user = kelimeGir.get()
    translated = GoogleTranslator(source='auto', target='tr').translate(text=user)
    sonuc['text']= translated

def cikis():
    pencere.destroy()

# Butonların stilini düzenleme
buton_stili = {'font': ('Helvetica', 12), 'bg': '#0066cc', 'fg': 'white', 'width': 15, 'height': 2, 'bd': 0}

Button = tk.Button(pencere, text="Dili Algıla", command=lambda: diliAlgila(data), **buton_stili)
Button.pack(pady=10)
Button1 = tk.Button(pencere, text="Dili Çevir", command=lambda: dilCevir(data), **buton_stili)
Button1.pack(pady=10)
cikisButonu = tk.Button(pencere, text="Çıkış", command=cikis, **buton_stili)
cikisButonu.pack(pady=10)
sonuc = tk.Label(pencere, fg="#FFFFFF", font=('Helvetica', 14, 'bold'), bg='#000000')
sonuc.pack(pady=10)

pencere.mainloop()
