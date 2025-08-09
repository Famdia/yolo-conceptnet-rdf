from flask import Flask, request, render_template, send_from_directory, jsonify, flash, url_for
import os
import uuid
import cv2
import requests
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from rdflib import Graph, Literal, Namespace, URIRef

# Initialisation de l'application Flask
app = Flask(__name__)

# Définition des dossiers de stockage des images
UPLOAD_FOLDER_ORIGINAL = 'uploads/images_originales'
UPLOAD_FOLDER_PROCESSED = 'uploads/images_traitees'
os.makedirs(UPLOAD_FOLDER_ORIGINAL, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_PROCESSED, exist_ok=True)
app.config['UPLOAD_FOLDER_ORIGINAL'] = UPLOAD_FOLDER_ORIGINAL
app.config['UPLOAD_FOLDER_PROCESSED'] = UPLOAD_FOLDER_PROCESSED
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = "supersecretkey"

# Chargement du modèle YOLOv8. J'ai choisi le yolov8s (small) au lieu du v8n (nano)
# car il est certes moins rapide mais il offre plus de précision
model = YOLO("yolov8s.pt")

# API ConceptNet
CONCEPTNET_API = "https://api.conceptnet.io/c/en/"

# Namespace RDF
EX = Namespace("http://ontologie/ia2s.fr/")

# ====== Endpoint : j'ai choisi de travailler avec GraphDB========
GRAPHDB_URL = "http://localhost:7200/repositories/projetYolo/statements"


# =========== Vérifie si le fichier a une extension autorisée=========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#=========== Enregistre l'image téléchargée sur le serveur avec un UUID pour éviter les conflits=======
def save_image(file):
    if not allowed_file(file.filename):
        flash("Type de fichier non autorisé!", "danger")
        return None, None
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER_ORIGINAL'], unique_filename)
    file.save(filepath)
    return filepath, unique_filename

#=========== Détecte les objets dans l'image avec YOLOv8 et sauvegarde l'image annotée==========
def detect_objects(image_path):
    results = model(image_path)
    detected_classes = set()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detected_classes.add(model.names[cls_id])

    annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER_PROCESSED'], os.path.basename(image_path))
    img = results[0].plot()
    cv2.imwrite(annotated_image_path, img)

    return list(detected_classes), annotated_image_path

#=========== Récupère uniquement les relations IsA, RelatedTo, UsedFor pour un label donné==========
def get_conceptnet_relations(label):
    url = f"{CONCEPTNET_API}{label.replace(' ', '_')}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la réponse est OK (200)
        relations = []
        for edge in response.json().get('edges', []):
            rel_type = edge['rel']['label']
            if rel_type in {"IsA", "CapableOf", "HasA", "UsedFor"}:
                start = edge['start']['label']
                end = edge['end']['label']
                relations.append((start, rel_type, end))
        return relations
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'appel à l'API ConceptNet : {e}")
        return []  # Retourne une liste vide en cas d'erreur


#=========== Insère les données dans GraphDB=======
def insert_into_graphdb(image_id, detections):
    g = Graph()
    g.bind("ex", EX)
    image_uri = URIRef(f"{EX}{image_id}")

    for label in detections:
        label_uri = URIRef(f"{EX}{label}")
        g.add((image_uri, EX.contains, label_uri))

        for start, relation, end in get_conceptnet_relations(label):
            g.add((label_uri, URIRef(f"{EX}{relation}"), Literal(end)))

    data = g.serialize(format='nt')
    query = f"INSERT DATA {{ {data} }}"
    requests.post(GRAPHDB_URL, data={'update': query}, headers={'Content-Type': 'application/x-www-form-urlencoded'})


# ============== NOS ROUTES ==================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('upload.html')  # Affiche la page d'upload

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            image_path, filename = save_image(file)
            if not image_path:
                return jsonify({"success": False, "message": "Erreur lors de l'enregistrement du fichier."}), 400

            labels, annotated_image_path = detect_objects(image_path)

            # Récupérer les relations ConceptNet pour chaque label détecté
            relations = {label: get_conceptnet_relations(label) for label in labels}

            return jsonify({
                "success": True,
                "message": "Image traitée avec succès!",
                "annotated_image_url": url_for('result_file', filename=os.path.basename(annotated_image_path)),
                "filename": filename,
                "labels": labels,
                "relations": relations   # Ajout des relations
            })

        return jsonify({"success": False, "message": "Aucun fichier ou format non autorisé."}), 400


@app.route('/store', methods=['POST'])
def store_results():
    data = request.get_json()
    filename = data.get("filename")
    labels = data.get("labels")

    if not filename or not labels:
        return jsonify({"success": False, "message": "Données manquantes"}), 400

    insert_into_graphdb(filename, labels)

    return jsonify({"success": True, "message": "Résultats stockés dans GraphDB !"})


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_ORIGINAL'], filename)


@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_PROCESSED'], filename)

@app.route('/search', methods=['GET', 'POST'])
def search_label():
    results = []  # Initialiser la liste des résultats des relations RDF
    images = []  # Initialiser la liste des résultats des images
    label = None  # Initialiser label à None pour les requêtes GET

    if request.method == 'POST':
        label = request.form['label']

        relation_query = f"""
        PREFIX ex: <http://ontologie/ia2s.fr/>
        SELECT ?s ?p ?o WHERE {{
            ?s ?p ?o .
            FILTER (regex(str(?s), "{label}", "i") || regex(str(?o), "{label}", "i"))
        }}
        """
        print("Requête pour rechercher les relations RDF envoyée à GraphDB:")
        print(relation_query)

        # Exécuter la requête SPARQL pour rechercher les relations
        response_relations = requests.get(GRAPHDB_URL, params={'query': relation_query}, headers={'Accept': 'application/rdf+xml'})

        print("Réponse brute de GraphDB pour les relations RDF:", response_relations.text)

        if response_relations.status_code == 200 and response_relations.text.strip():
            try:
                graph_relations = Graph()
                graph_relations.parse(data=response_relations.text, format="xml")

                # Extraire les relations RDF
                for subj, pred, obj in graph_relations:
                    subj_str = str(subj)
                    obj_str = str(obj)
                    pred_str = str(pred)

                    # Ajouter les résultats des relations liées au label
                    if label.lower() in subj_str.lower():  # Vérification que le sujet contient le label
                        results.append({
                            'subject': subj_str,
                            'predicate': pred_str,
                            'object': obj_str
                        })

                    # Si l'objet contient l'image et que le prédicat est "contains"
                    if label.lower() in obj_str.lower() and pred_str == "http://ontologie/ia2s.fr/contains":
                        image_filename = subj_str.split('/')[-1]  # Extraire le nom du fichier à partir de l'URL
                        images.append(image_filename)

                print("Relations récupérées:", results)
                print("Images récupérées:", images)
            except Exception as e:
                print("Erreur lors du décodage RDF pour les relations:", e)

    return render_template('search.html', label=label, images=images, results=results)


# Désactiver le cache côté Flask
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True)
