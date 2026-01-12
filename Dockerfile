# 1️⃣ Image de base Python stable
FROM python:3.10-slim

# 2️⃣ Installer les dépendances système utiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    cmake \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Définir le répertoire de travail
WORKDIR /app

# 4️⃣ Copier le fichier requirements.txt et installer les packages
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copier le reste du projet
COPY . .

# 6️⃣ Exposer le port Streamlit
EXPOSE 8501

# 7️⃣ Commande par défaut pour lancer l'application
CMD ["streamlit", "run", "Lamia/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
