from flask import Flask , render_template , request
from werkzeug.utils import secure_filename #API de traitement texte des noms de fichiers (ex: image du 1.img ) --> filename(image_du_1.img)
from matplotlib import pyplot
from PIL import Image  # Bibliothèque de traitement d'image qui donne l'accès aux données cotenues dans une image.
import numpy as np
import pickle
import os

app = Flask(__name__)
app.debug = True

@app.route("/")
def upload():
    return render_template('index.html') 

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
   
# Créer un répertoire temporaire pour sauvegarder le fichier dedans. 
   uploads_dir = os.path.join(r'C:\Users\utilisateur\Desktop\IA\learningCode\Projects_Simplon\MNIST\MNIST-Web-App', 'static')
   
# Charge le model de ML. 
   model = pickle.load(open('modelKNNnor.pkl', 'rb'))
   
# Vérifie la condition de l'upload d'image et que l'utilisateur upload une image. 
   if request.method == 'POST':  
      
# Récupère la requete ainsi que l'image dans une variable et l'enregistre en local. 
      f = request.files['file']
      
# Enregistre l'image dans le dossier, applique le traitement de texte sur le nom de l'image. 
      f.save(os.path.join(uploads_dir , secure_filename(f.filename)))  


# change le mode de l’image. L représente 256 nuances de gris "transforme les images couleurs en gris ". 
      img = Image.open(f).convert('L') 
      
# Redimensionne l'image dans le bon format et applique l'option LANCZOS qui permet de rendre l'image plus clair (Calculate the output pixel value using a high-quality Lanczos filter (a truncated sinc) on all pixels that may contribute to the output value. )
      img = img.resize((28,28) , Image.LANCZOS)
      #print(img)
      
# Transforme l'image sous forme d'array et reshape  
      data = ((np.asarray(img))/255.0)

# On applique le modèle sur notre image qui vient d'etre reshape dans le format demandé par le modèle soit un seul vecteur contenant l'ensemble des infos. 
      pred = model.predict(data.reshape(1,-1)) 
      print(pred)

      return render_template('prediction.html' , out = str(pred[0]) , im = f.filename)



if __name__ == '__main__':
      app.run(debug = True)