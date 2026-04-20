# Méthodes Classiques (OpenCV)

Ce dossier contient nos premiers essais pour détecter le BlueROV en utilisant des algorithmes classiques de traitement d'images (OpenCV). Nous avons testé ces méthodes avant de nous rendre compte qu'il fallait passer par une IA (YOLO) pour contrer les reflets de l'eau.

## Prérequis
Il faut juste installer OpenCV, Numpy et MSS (qui sert à filmer l'écran de l'ordinateur en direct).
Tapez ça dans le terminal :
```bash
pip install opencv-python numpy mss
```

## Les 4 codes à tester

### 1. `HSV.py` (Filtrage par couleur)
* **Principe :** Le code filme une partie de votre écran. Il cherche la plus grosse masse sombre/noire et calcule son barycentre.
* **Comment l'utiliser :** Lancez le code, placez une vidéo du robot dans la zone de capture de votre écran. Pour arrêter, appuyez sur `q`.
* **Pourquoi on l'a abandonné :** Sous l'eau, il confond le robot avec son ombre sur le fond du bassin ou des reflets sombres.

### 2. `CAMSHIFT.py` (Suivi par histogramme)
* **Principe :** Il mémorise l'empreinte de couleur du robot et essaie de suivre cette masse de couleur.
* **Comment l'utiliser :** Modifiez la ligne `nom_video = ...` avec votre fichier. Au lancement, la vidéo se met en pause. Entourez le robot avec la souris, puis appuyez sur **ESPACE**.
* **Pourquoi on l'a abandonné :** En plongeant, l'eau absorbe la lumière et change la couleur du robot. Le code perd la cible presque immédiatement.

### 3. `ORB.py` (Points d'intérêts)
* **Principe :** Il compare chaque image de la vidéo avec des images de référence du robot en CAO. S'il trouve assez d'angles et de points communs, il encadre le robot.
* **Comment l'utiliser :** Assurez-vous d'avoir un dossier nommé `imagesCAO2` contenant des photos du robot au même endroit que le script.
* **Pourquoi on l'a abandonné :** La coque du robot est lisse et l'eau "lisse" les contrastes. L'algo ne trouve pas de points d'accroche et attrape les reflets à la place.

### 4. `HYBRIDE.py` (Notre tentative finale)
* **Principe :** C'est une combinaison de nos codes. Il utilise `HSV` pour repérer un candidat noir, il vérifie avec `ORB` si c'est bien le robot, et si oui, il lance un tracker `MIL` d'OpenCV pour le suivre de manière fluide.
* **Comment l'utiliser :** Comme le premier code, il filme l'écran en direct. (N'oubliez pas le dossier `imagesCAO2`).

---
**Bilan :** Tous ces codes fonctionnent correctement dans l'air, mais se font piéger par les perturbations visuelles de l'eau (reflets, changements de couleur, perte de contraste). 
Pour le bassin d'essai, passez directement au code principal du projet qui utilise YOLOv8.
