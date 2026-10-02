
import tensorflow as tf
import numpy as np
import librosa
import os

from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# =========================================================
# PARAMÈTRES
# =========================================================

TAUX_ECHANTILLONNAGE = 16000
NOMBRE_MFCC = 40
LONGUEUR_MFCC = 200

BATCH_SIZE = 16
EPOCHS = 30

# Seuil utilisé pour décider si une voix est suffisamment
# proche d'une personne connue
SEUIL_CONFIANCE = 0.70


# =========================================================
# CHARGEMENT DU DATASET
# =========================================================

dataset_path = (
    "C:/Users/ASUS/PycharmProjects/"
    "Projet_Identification_Voix/Dataset"
)


# =========================================================
# VÉRIFICATION DU DATASET
# =========================================================

if not os.path.exists(dataset_path):

    print("ERREUR : le dossier Dataset n'existe pas.")
    print("Chemin utilisé :", dataset_path)
    exit()


# =========================================================
# RÉCUPÉRATION DES PERSONNES
# =========================================================

personnes = sorted([
    dossier
    for dossier in os.listdir(dataset_path)
    if os.path.isdir(
        os.path.join(dataset_path, dossier)
    )
])


print("\n===== PERSONNES =====")

print(personnes)

print(
    "Nombre de classes :",
    len(personnes)
)


# =========================================================
# VÉRIFICATION DE LA CLASSE AUTRES
# =========================================================

if "Autres" not in personnes:

    print("\nATTENTION : le dossier 'Autres' n'existe pas.")
    print("Crée le dossier :")
    print(
        os.path.join(
            dataset_path,
            "Autres"
        )
    )

    print(
        "\nAjoute plusieurs voix de personnes "
        "qui ne sont pas Personne1, Personne2, Personne3..."
    )

    exit()


# =========================================================
# VÉRIFICATION DU NOMBRE DE CLASSES
# =========================================================

if len(personnes) < 2:

    print(
        "\nERREUR : il faut au moins 2 classes."
    )

    exit()


# =========================================================
# LECTURE DU FICHIER FLAC
# =========================================================

def lire_audio(
        chemin_audio,
        sr=TAUX_ECHANTILLONNAGE
):

    try:

        audio, sr = librosa.load(

            chemin_audio,

            sr=sr,

            mono=True
        )

        return audio, sr

    except Exception as erreur:

        print(
            "\nERREUR pendant la lecture :"
        )

        print(
            chemin_audio
        )

        print(
            erreur
        )

        return None, None


# =========================================================
# EXTRACTION DES MFCC
# =========================================================

def extraire_mfcc(chemin_audio):

    audio, sr = lire_audio(
        chemin_audio
    )


    if audio is None:

        return None


    if len(audio) == 0:

        return None


    # =====================================================
    # EXTRACTION DES MFCC
    # =====================================================

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sr,

        n_mfcc=NOMBRE_MFCC
    )


    # =====================================================
    # NORMALISATION DE LA LONGUEUR
    # =====================================================

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


    return mfcc


# =========================================================
# PRÉPARATION DES DONNÉES
# =========================================================

X = []
y = []


print("\n===== CHARGEMENT DES AUDIOS =====")


for numero_personne, personne in enumerate(personnes):

    dossier_personne = os.path.join(

        dataset_path,

        personne
    )


    nombre_fichiers = 0


    for fichier in os.listdir(
        dossier_personne
    ):


        # -------------------------------------------------
        # UNIQUEMENT LES FICHIERS FLAC
        # -------------------------------------------------

        if fichier.lower().endswith(".flac"):

            chemin = os.path.join(

                dossier_personne,

                fichier
            )


            print(
                "Traitement :",
                personne,
                "->",
                fichier
            )


            mfcc = extraire_mfcc(
                chemin
            )


            if mfcc is not None:

                X.append(
                    mfcc
                )

                y.append(
                    numero_personne
                )

                nombre_fichiers += 1


    print(
        "Nombre de fichiers pour",
        personne,
        ":",
        nombre_fichiers
    )


# =========================================================
# CONVERSION NUMPY
# =========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y
)


print("\n===== DONNÉES =====")

print(
    "Taille de X :",
    X.shape
)

print(
    "Taille de y :",
    y.shape
)


# =========================================================
# VÉRIFICATION
# =========================================================

if len(X) == 0:

    print(
        "\nERREUR : aucun fichier FLAC n'a été trouvé."
    )

    print(
        "Vérifie que les fichiers .flac sont "
        "directement dans les dossiers."
    )

    exit()


# =========================================================
# NOMBRE DE FICHIERS PAR CLASSE
# =========================================================

print("\n===== NOMBRE DE FICHIERS PAR CLASSE =====")


for numero_personne, personne in enumerate(personnes):

    nombre = np.sum(
        y == numero_personne
    )

    print(
        personne,
        ":",
        nombre,
        "fichier(s)"
    )


# =========================================================
# NORMALISATION
# =========================================================

moyenne = np.mean(X)

ecart_type = np.std(X)


X = (
    X - moyenne
) / (
    ecart_type + 1e-8
)


# =========================================================
# CONVERSION DES LABELS
# =========================================================

y_cat = to_categorical(

    y,

    num_classes=len(personnes)
)


# =========================================================
# SÉPARATION ENTRAÎNEMENT / TEST
# =========================================================

print("\n===== SÉPARATION ENTRAÎNEMENT / TEST =====")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_cat,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print(
    "Nombre entraînement :",
    len(X_train)
)

print(
    "Nombre test :",
    len(X_test)
)


# =========================================================
# TRANSFORMATION DES MFCC
# =========================================================

X_train = X_train.reshape(

    X_train.shape[0],

    -1
)


X_test = X_test.reshape(

    X_test.shape[0],

    -1
)


# =========================================================
# MODÈLE
# =========================================================

print("\n===== CRÉATION DU MODÈLE =====")


modele = tf.keras.Sequential([

    layers.Dense(

        128,

        activation="relu",

        input_shape=(
            NOMBRE_MFCC * LONGUEUR_MFCC,
        )
    ),

    layers.Dropout(
        0.3
    ),

    layers.Dense(

        64,

        activation="relu"
    ),

    layers.Dense(

        32,

        activation="relu"
    ),

    layers.Dense(

        len(personnes),

        activation="softmax"
    )
])


# =========================================================
# COMPILATION
# =========================================================

modele.compile(

    optimizer=tf.keras.optimizers.SGD(

        learning_rate=0.01,

        momentum=0.9
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# =========================================================
# RÉSUMÉ
# =========================================================

print("\n===== RÉSUMÉ DU MODÈLE =====")

modele.summary()


# =========================================================
# ENTRAÎNEMENT
# =========================================================

print("\n===== ENTRAÎNEMENT =====")


historique = modele.fit(

    X_train,

    y_train,

    validation_data=(

        X_test,

        y_test
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE
)


# =========================================================
# ÉVALUATION
# =========================================================

print("\n===== ÉVALUATION =====")


loss, accuracy = modele.evaluate(

    X_test,

    y_test
)


print(
    f"\nLoss     : {loss:.4f}"
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Accuracy : {accuracy * 100:.2f} %"
)


# =========================================================
# TEST D'UN NOUVEAU FICHIER FLAC
# =========================================================

print("\n===== RECONNAISSANCE =====")


fichier_test = (
    "C:/Users/ASUS/Documents/ReseauNeurone/"
    "Nouveau dossier/LibriSpeech/dev-clean/"
    "174/50561/174-50561-0013.flac"
)


# =========================================================
# VÉRIFICATION DU FICHIER
# =========================================================

if not os.path.exists(fichier_test):

    print(
        "ERREUR : le fichier test n'existe pas."
    )

    print(
        "Chemin :",
        fichier_test
    )

    exit()


# =========================================================
# EXTRACTION MFCC
# =========================================================

mfcc_test = extraire_mfcc(

    fichier_test
)


if mfcc_test is None:

    print(
        "ERREUR : impossible de traiter "
        "le fichier test."
    )

    exit()


# =========================================================
# MÊME NORMALISATION QUE L'ENTRAÎNEMENT
# =========================================================

mfcc_test = (

    mfcc_test - moyenne

) / (

    ecart_type + 1e-8
)


# =========================================================
# TRANSFORMATION EN VECTEUR
# =========================================================

mfcc_test = mfcc_test.reshape(

    1,

    -1
)


# =========================================================
# PRÉDICTION
# =========================================================

prediction = modele.predict(

    mfcc_test,

    verbose=0
)


# =========================================================
# CLASSE AVEC LA PLUS GRANDE PROBABILITÉ
# =========================================================

indice = np.argmax(

    prediction[0]
)


confiance = prediction[0][indice]


# =========================================================
# DÉCISION : PERSONNE CONNUE OU AUTRES
# =========================================================

if confiance < SEUIL_CONFIANCE:

    personne_reconnue = "Autres"

    print(
        "\n⚠️ VOIX INCONNUE"
    )

    print(
        "La voix est classée comme : AUTRES"
    )

else:

    personne_reconnue = personnes[indice]

    print(
        "\nPersonne reconnue :",
        personne_reconnue
    )


print(
    "Confiance du modèle :",
    round(
        confiance * 100,
        2
    ),
    "%"
)


# =========================================================
# AFFICHAGE DE TOUTES LES PROBABILITÉS
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
# SAUVEGARDE DU MODÈLE
# =========================================================

chemin_modele = (
    "C:/Users/ASUS/PycharmProjects/"
    "Projet_Identification_Voix/"
    "modele_reconnaissance_voix.h5"
)


modele.save(

    chemin_modele
)


print(
    "\nModèle sauvegardé :"
)

print(
    chemin_modele
)


# =========================================================
# SAUVEGARDE DES NOMS DES PERSONNES
# =========================================================

chemin_personnes = (
    "C:/Users/ASUS/PycharmProjects/"
    "Projet_Identification_Voix/"
    "personnes.npy"
)


np.save(

    chemin_personnes,

    np.array(personnes)
)


print(
    "\nNoms des personnes sauvegardés."
)


print(
    "\n===== FIN DU PROGRAMME ====="
)
```
