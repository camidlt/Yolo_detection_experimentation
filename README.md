# Détection d'Objets avec YOLOv8 et Streamlit

Cette application web permet de détecter des objets dans des vidéos YouTube en utilisant YOLOv8 (You Only Look Once) et Streamlit.

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation

1. Clonez ce dépôt
2. Créez un environnement virtuel :
```bash
python -m venv venv
```

3. Activez l'environnement virtuel :
- Windows :
```bash
venv\Scripts\activate
```
- Linux/Mac :
```bash
source venv/bin/activate
```

4. Installez les dépendances :
```bash
pip install -r requirements.txt
```

5. Le modèle YOLOv8 sera automatiquement téléchargé lors de la première exécution de l'application.

## Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8501)
3. Collez l'URL d'une vidéo YouTube
4. Attendez le traitement de la vidéo
5. Visualisez les résultats de la détection d'objets

## Notes

- Assurez-vous d'avoir une connexion internet stable
- Le traitement des vidéos peut prendre du temps selon leur durée
- Respectez les droits d'auteur et les conditions d'utilisation de YouTube
- YOLOv8 est plus rapide et plus précis que les versions précédentes de YOLO 