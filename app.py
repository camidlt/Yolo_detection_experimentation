import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict

# Configuration de la page Streamlit
st.set_page_config(page_title="Détection d'Objets", page_icon="🎥", layout="wide")
st.title("🎥 Détection d'Objets dans les Vidéos")
st.markdown("Téléchargez une vidéo MP4 pour détecter les objets en temps réel")

# Fonction pour charger le modèle YOLO
@st.cache_resource
def load_yolo_model():
    # Charger le modèle YOLO pré-entraîné
    model = YOLO("yolo11n.pt")  # ou "yolo11n.pt" si vous utilisez YOLO11
    return model

# Interface utilisateur
st.markdown("### 📹 Téléchargement de la vidéo")
uploaded_file = st.file_uploader("Choisir une vidéo", type=['mp4'])

# Paramètres de traitement
if uploaded_file is not None:
    st.markdown("### ⚙️ Paramètres de traitement")

    col1, col2 = st.columns(2)
    with col1:
        frame_skip = st.slider(
            "🎯 Fréquence de traitement (frames à sauter)",
            1, 30, 5,
            help="Plus la valeur est élevée, plus le traitement sera rapide mais moins fluide"
        )
    with col2:
        max_duration = st.number_input(
            "⏱️ Durée maximale à traiter (secondes)",
            min_value=5,
            max_value=300000000,
            value=60,
            help="Limite la durée de la vidéo à traiter"
        )
    
    # Calcul automatique des statistiques de sortie (estimations)
    st.markdown("### 📊 Estimation du traitement")
    
    # Estimations basées sur des valeurs standards
    estimated_fps = 30  # FPS standard pour estimation
    estimated_duration = max_duration  # Durée à traiter
    
    # Calculs d'estimation
    estimated_frames_to_process = int(estimated_duration * estimated_fps)
    estimated_processed_frames = estimated_frames_to_process // frame_skip
    estimated_output_duration = estimated_duration / frame_skip
    
    # Métriques de traitement estimées
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "⏱️ Durée de sortie estimée", 
            f"{estimated_output_duration:.1f}s",
            delta=f"{estimated_output_duration - estimated_duration:.1f}s"
        )
    with col2:
        st.metric(
            "🎬 Frames estimées", 
            f"{estimated_processed_frames:,}",
            delta=f"-{estimated_frames_to_process - estimated_processed_frames:,}"
        )
    with col3:
        st.metric(
            "📈 Compression", 
            f"{frame_skip}:1",
            help="Ratio de compression des frames"
        )
    with col4:
        efficiency = (estimated_processed_frames / estimated_frames_to_process) * 100
        st.metric(
            "⚡ Efficacité", 
            f"{efficiency:.1f}%",
            help="Pourcentage de frames traitées"
        )

# Bouton de traitement
if uploaded_file is not None:
    st.markdown("### 🚀 Lancement du traitement")

    if st.button("🎬 Démarrer la détection", type="primary", use_container_width=True):
        try:
            # Sauvegarde temporaire de la vidéo
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()

            # Chargement du modèle YOLO
            with st.spinner("🤖 Chargement du modèle YOLO..."):
                model = load_yolo_model()

            # Interface de traitement
            st.markdown("### 🎥 Traitement en cours")

            # Conteneurs pour l'affichage
            col1, col2 = st.columns([2, 1])

            with col1:
                stframe = st.empty()

            with col2:
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_container = st.empty()

            # Initialiser un dictionnaire pour compter les objets uniques
            object_counts = {}

            # Traitement de la vidéo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Impossible d'ouvrir la vidéo")
                st.stop()

            # Calcul du nombre total de frames à traiter
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_duration * fps))
            frame_count = 0

            st.info(f"🎬 Traitement de {total_frames:,} frames à {fps:.1f} FPS")

            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Traitement selon la fréquence sélectionnée
                if frame_count % frame_skip == 0:
                    # Redimensionner la frame si elle est trop grande
                    height, width = frame.shape[:2]
                    max_dimension = 640
                    if max(height, width) > max_dimension:
                        scale = max_dimension / max(height, width)
                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                    # Détection avec YOLO
                    results = model.track(frame, persist=True)  # Utiliser track pour suivre les objets

                    # Dessiner les résultats sur la frame
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Coordonnées de la boîte
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # Classe et confiance
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if box.id is not None else None

                            # Dessiner la boîte
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Ajouter le label
                            label = f"{result.names[cls]} {conf:.2f}"
                            if track_id is not None:
                                label += f" ID: {track_id}"
                                # Mettre à jour le compte des objets
                                if track_id in object_counts:
                                    object_counts[track_id] = (object_counts[track_id][0], object_counts[track_id][1] + 1)
                                else:
                                    object_counts[track_id] = (result.names[cls], 1)

                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    stframe.image(frame, channels="BGR", width=640)

                # Mise à jour de la barre de progression
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"📊 Progression: {int(progress * 100)}% ({frame_count:,}/{total_frames:,} frames)")

                # Mise à jour des statistiques en temps réel
                with stats_container.container():
                    st.metric("Frame actuelle", f"{frame_count:,}")
                    if fps > 0:
                        st.metric("Temps restant", f"{((total_frames - frame_count) / fps):.1f}s")

                frame_count += 1

            cap.release()
            processed_frames = frame_count // frame_skip

            # Nettoyage
            os.remove(video_path)


            class_counts = defaultdict(int)
            # Parcourir le dictionnaire object_counts et additionner les apparitions par classe
            for obj_id, (cls_name, count) in object_counts.items():
                class_counts[cls_name] += count

            # Afficher les résultats du comptage
            st.markdown("### Résultats du comptage")
            # Afficher le nombre total d'apparitions pour chaque classe
            for cls_name, total_count in class_counts.items():
                st.write(f"{cls_name}: {total_count}")

            # Message de succès
            st.success(f"🎉 Traitement terminé! {processed_frames:,} frames traitées avec succès.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue: {str(e)}")
            if 'video_path' in locals():
                try:
                    os.remove(video_path)
                except:
                    pass

# Pied de page
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔧 Développé avec Streamlit et YOLO</p>
        <p><small>💡 Astuce: Utilisez des vidéos claires et bien cadrées pour de meilleurs résultats</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
