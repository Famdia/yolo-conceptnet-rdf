# Reconnaissance sémantique des images et stockage dans un graphe de connaissances OWL RDF

Ce projet permet aux utilisateurs de :

- Télécharger des images via une interface web. 
- Détecter automatiquement les objets présents dans ces images grâce à YOLOv8. 
- Enrichir les objets détectés avec des relations issues de ConceptNet. 
- Générer et stocker les informations sous forme de triplets RDF dans GraphDB. 
- Rechercher des labels et récupérer les images et relations associées.

## Technologies utilisées

* Flask : Backend et gestion des routes.
* Bootstrap : Mise en page et design côté front-end.
* JavaScript (Fetch API) : Interaction dynamique côté client.
* YOLOv8s : Détection d’objets.
* ConceptNet : Enrichissement sémantique des objets détectés.
* GraphDB : Stockage et interrogation RDF.

## Installation et utilisation

**1. Prérequis**

* Python 3.8+ installé sur votre machine
* pip (gestionnaire de paquets Python)
* Modèle YOLOv8s (yolov8s.pt) disponible (téléchargé automatiquement par ultralytics si absent)

**2. Cloner le projet**

```
git clone https://github.com/Famdia/yolo-conceptnet-rdf.git
```

**3. Installer les dépendances**

Je vous recommande fortement de créer un environnement virtuel 

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

Installer les dépendances contenues dans le fichier requirements.txt

```
pip install -r requirements.txt
```

**4. Lancer l'application**

```
python application.py
```

Pour accéder à l'interface web, rendez vous à l'adresse http://localhost:5000
