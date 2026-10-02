
import tensorflow as tf
import numpy as np
import librosa
import os


# =========================================================
# PARAMÈTRES
# =========================================================

TAUX_ECHANTILLONNAGE = 16000
NOMBRE_MFCC = 40
LONGUEUR_MFCC = 200

SEUIL_RECONNAISSANCE = 0.70


# =========================================================
# CHEMINS
# =========================================================

chemin_modele = (
    "C:/Users/ASUS/PycharmProjects/"
    "Projet_Identification_Voix/"
    "modele_reconnaissance_voix.h5"
)

chemin_personnes = (
    "C:/Users/ASUS/PycharmProjects/"
    "Projet_Identification_Voix/"
    "personnes.npy"
)


# =========================================================
# FICHIER AUDIO À TESTER
# =========================================================

fichier_test = (
    "C:/Users/ASUS/Documents/ReseauNeurone/Nouveau dossier/LibriSpeech/dev-clean/251/137823/251-137823-0008.flac"
)


# =========================================================
# VÉRIFICATION DU MODÈLE
# =========================================================

if not os.path.exists(chemin_modele):

    print("ERREUR : modèle introuvable.")

    print(
        "Chemin :",
        chemin_modele
    )

    exit()


# =========================================================
# VÉRIFICATION DES NOMS
# =========================================================

if not os.path.exists(chemin_personnes):

    print("ERREUR : fichier personnes.npy introuvable.")

    exit()


# =========================================================
# VÉRIFICATION DE L'AUDIO
# =========================================================

if not os.path.exists(fichier_test):

    print("ERREUR : fichier audio introuvable.")

    print(
        "Chemin :",
        fichier_test
    )

    exit()


# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================

print("\n===== CHARGEMENT DU MODÈLE =====")

modele = tf.keras.models.load_model(
    chemin_modele
)

print("Modèle chargé avec succès.")


# =========================================================
# CHARGEMENT DES NOMS DES PERSONNES
# =========================================================

personnes = np.load(
    chemin_personnes,
    allow_pickle=True
)


personnes = personnes.tolist()


print(
    "Classes :",
    personnes
)


# =========================================================
# LECTURE DE L'AUDIO
# =========================================================

print("\n===== LECTURE DE L'AUDIO =====")

audio, sr = librosa.load(
    fichier_test,
    sr=TAUX_ECHANTILLONNAGE,
    mono=True
)


print(
    "Audio chargé."
)


# =========================================================
# EXTRACTION DES MFCC
# =========================================================

print("\n===== EXTRACTION DES MFCC =====")

mfcc = librosa.feature.mfcc(

    y=audio,

    sr=sr,

    n_mfcc=NOMBRE_MFCC
)


# =========================================================
# NORMALISATION DE LA LONGUEUR
# =========================================================

if mfcc.shape[1] < LONGUEUR_MFCC:

    mfcc = np.pad(

        mfcc,

        (
            (0, 0),

            (
                0,
                LONGUEUR_MFCC - mfcc.shape[1]
            )
        ),

        mode="constant"
    )

else:

    mfcc = mfcc[
        :,
        :LONGUEUR_MFCC
    ]


# =========================================================
# NORMALISATION
# =========================================================
# IMPORTANT :
# Le modèle d'entraînement utilisait une normalisation.
# Ici on fait une normalisation du fichier test.

moyenne = np.mean(mfcc)

ecart_type = np.std(mfcc)


mfcc = (

    mfcc - moyenne

) / (

    ecart_type + 1e-8
)


# =========================================================
# TRANSFORMATION EN VECTEUR
# =========================================================

mfcc = mfcc.reshape(
    1,
    -1
)


# =========================================================
# PRÉDICTION
# =========================================================

print("\n===== PRÉDICTION =====")

prediction = modele.predict(
    mfcc,
    verbose=0
)


# =========================================================
# AFFICHAGE DES PROBABILITÉS
# =========================================================

print("\n===== PROBABILITÉS =====")


for i, personne in enumerate(personnes):

    print(

        personne,

        ":",

        round(
            prediction[0][i] * 100,
            2
        ),

        "%"
    )


# =========================================================
# RECHERCHE DE LA PROBABILITÉ MAXIMALE
# =========================================================

indice = np.argmax(
    prediction[0]
)


classe_predite = personnes[indice]

confiance = prediction[0][indice]


# =========================================================
# DÉCISION AUTRES
# =========================================================

if classe_predite == "Autres":

    personne_reconnue = "Autres"


elif confiance >= SEUIL_RECONNAISSANCE:

    personne_reconnue = classe_predite


else:

    personne_reconnue = "Autres"


# =========================================================
# RÉSULTAT
# =========================================================

print("\n==============================")

print(
    "PERSONNE RECONNUE :",
    personne_reconnue
)

print(
    "CONFIANCE :",
    round(
        confiance * 100,
        2
    ),
    "%"
)

print(
    "SEUIL :",
    SEUIL_RECONNAISSANCE * 100,
    "%"
)

print("==============================")

