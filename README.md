# Méthodes Classiques de Détection (OpenCV)

Ce dossier contient nos premiers essais pour détecter le BlueROV2 en utilisant des algorithmes classiques de traitement d'images (OpenCV). Nous avons testé ces méthodes avant de nous rendre compte qu'il fallait passer par une IA (YOLO) pour contrer les reflets de l'eau.

> 📁 Ce repo est une **archive d'exploration**. Pour la détection finale qui fonctionne vraiment en bassin, rendez-vous sur le [repo principal du projet](lien_vers_repo_principal).

---

## Structure du projet

```
Methodes-Classiques-OpenCv-main/
├── HSV.py          → Détection par filtrage de couleur (capture d'écran en direct)
├── CAMSHIFT.py     → Suivi par histogramme de couleur (sur fichier vidéo)
├── ORB.py          → Détection par points d'intérêt / matching CAO (sur fichier vidéo)
├── HYBRIDE.py      → Combinaison HSV + ORB + tracker MIL (capture d'écran en direct)
└── imagesCAO2/     → 6 renders CAO du BlueROV2 sous différents angles (références pour ORB)
```

---

## Prérequis

- **Python 3.8+**
- Les bibliothèques suivantes :

```bash
pip install opencv-python numpy mss
```

`mss` sert uniquement à capturer l'écran en direct (utilisé par `HSV.py` et `HYBRIDE.py`). Si vous testez uniquement `CAMSHIFT.py` ou `ORB.py`, vous n'en avez pas besoin.

---

## Les 4 codes à tester

### 1. `HSV.py` — Filtrage par couleur

* **Principe :** Le code filme une zone fixe de votre écran en direct. Il cherche la plus grosse masse sombre/noire dans cette zone et calcule son barycentre à l'aide des moments spatiaux d'OpenCV.

* **Comment l'utiliser :**
  1. Lancez le code, puis placez une vidéo du robot dans la zone de capture de votre écran.
  2. Deux fenêtres s'ouvrent : la vue caméra avec la bounding box, et le masque HSV pour débugger.
  3. Pour arrêter, appuyez sur `q`.

  > ⚙️ **À adapter :** La zone de capture est définie ligne 5 du script (`zone_capture`). Modifiez `top`, `left`, `width` et `height` selon votre résolution d'écran.

* **Pourquoi on l'a abandonné :** Sous l'eau, le robot est loin d'être le seul objet sombre à l'écran. Il confond facilement le châssis noir avec son ombre sur le fond du bassin ou avec des reflets sombres en surface.

---

### 2. `CAMSHIFT.py` — Suivi par histogramme

* **Principe :** Il mémorise l'empreinte de couleur du robot (son histogramme HSV) sur une zone que vous sélectionnez à la souris, puis essaie de retrouver et de suivre cette masse de couleur dans toute la vidéo.

* **Comment l'utiliser :**
  1. Modifiez la ligne `nom_video = 'BillyBassinX2.mov'` en haut du script avec le chemin vers votre propre fichier vidéo.
  2. Lancez le code. La vidéo se met en pause sur la première image.
  3. Tracez un rectangle autour du robot avec la souris, puis appuyez sur **ESPACE** pour démarrer le suivi.
  4. Pour arrêter, appuyez sur `q`.

* **Pourquoi on l'a abandonné :** En plongeant, l'eau absorbe la lumière et change radicalement la couleur perçue du robot. L'histogramme enregistré en surface ne correspond plus du tout à ce que l'algo voit en profondeur — il perd la cible presque immédiatement.

---

### 3. `ORB.py` — Détection par points d'intérêt

* **Principe :** Il compare chaque image de la vidéo avec les 6 images de référence CAO du robot (stockées dans `imagesCAO2/`). Pour chaque image, il extrait des points d'intérêt (coins, angles) et leurs descripteurs avec ORB, puis les compare via un BFMatcher. S'il trouve plus de 15 correspondances valides (filtre de Lowe + RANSAC), il projette le contour du robot sur la vidéo.

* **Comment l'utiliser :**
  1. Vérifiez que le dossier `imagesCAO2/` est bien présent au même endroit que le script (il contient déjà les 6 références du BlueROV2).
  2. Modifiez la ligne `cv2.VideoCapture('BillyBassinX2.mov')` avec le chemin vers votre fichier vidéo.
  3. Lancez le code.

  > 💡 **Pour un autre robot :** remplacez les images du dossier `imagesCAO2/` par des vues de votre propre objet (plusieurs angles, fond neutre de préférence).

* **Pourquoi on l'a abandonné :** La coque du BlueROV2 est lisse et uniforme, et l'eau "lisse" encore plus les contrastes. ORB ne trouve pas de points d'accroche sur une surface sans texture, et finit par matcher les reflets à la place du robot.

---

### 4. `HYBRIDE.py` — Notre tentative finale

* **Principe :** On a combiné les trois approches précédentes pour limiter leurs défauts respectifs. Le pipeline fonctionne en deux phases :
  - **Phase Recherche :** `HSV` repère un candidat sombre, puis `ORB` vérifie que ce candidat ressemble bien aux images de référence (seuil : 100 correspondances). Un filtre de distance évite que la détection saute sur une ombre lointaine.
  - **Phase Suivi :** Une fois le robot validé, un tracker `MIL` d'OpenCV prend le relais pour un suivi fluide. Toutes les 2 secondes, le système repasse en phase Recherche pour vérifier qu'il suit encore le bon objet.

* **Comment l'utiliser :**
  1. Vérifiez que le dossier `imagesCAO2/` est bien présent.
  2. Lancez le code. Placez la vidéo du robot dans la zone de capture de votre écran.
  3. Le système affiche "Recherche en cours..." jusqu'à valider le robot, puis bascule automatiquement en mode suivi.
  4. Pour arrêter, appuyez sur `q`.

  > ⚙️ **À adapter :**
  > - La zone de capture est définie ligne 33 (`capture_zone`). Ajustez-la à votre écran.
  > - Le seuil de validation ORB est `max_good_matches > 100` (ligne ~100). Si la détection ne démarre jamais, baissez ce seuil à 30–50.

---

## Bilan

Tous ces codes fonctionnent correctement dans l'air, mais se font piéger par les perturbations visuelles de l'eau (reflets, changements de couleur, perte de contraste). La conclusion est simple : les méthodes classiques ne sont pas robustes pour ce cas d'usage.

Pour la détection en bassin, nous sommes passés à **YOLOv8**, qui gère nativement les cas difficiles grâce à son apprentissage sur des images variées. Le repo principal est ici → [lien_vers_repo_principal].

---

*Projet réalisé dans le cadre du module de vision par ordinateur — SeaTech, 2025-2026.*
