import cv2
import numpy as np

# Chargement de la vidéo
nom_video = 'BillyBassinX2.mov' 
cap = cv2.VideoCapture(nom_video)

if not cap.isOpened():
    print("Erreur vidéo")
    exit()

# Lecture de la première frame
ret, frame = cap.read()
frame = cv2.resize(frame, (800, 600))

# Sélection manuelle du robot à la souris
print("Tracez une boîte autour du robot puis appuyez sur ESPACE.")
roi_box = cv2.selectROI("Initialisation", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Initialisation")

x, y, w, h = int(roi_box[0]), int(roi_box[1]), int(roi_box[2]), int(roi_box[3])
fenetre_suivi = (x, y, w, h)

# Récupération de la zone sélectionnée (ROI)
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Masque pour ignorer les reflets et les ombres
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# Calcul de l'histogramme de couleur du robot
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Critères d'arrêt de l'algorithme
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Boucle vidéo
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (800, 600))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Cherche les pixels qui correspondent à la couleur du robot
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # CamShift met à jour la position et la taille de la boîte
    ret_camshift, fenetre_suivi = cv2.CamShift(dst, fenetre_suivi, term_crit)

    # Dessin de la boîte inclinée
    pts = cv2.boxPoints(ret_camshift)
    pts = np.int32(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    
    # Affichage du centre
    centre_x = int(ret_camshift[0][0])
    centre_y = int(ret_camshift[0][1])
    cv2.circle(frame, (centre_x, centre_y), 5, (0, 0, 255), -1)

    # Affichage vidéo
    cv2.imshow("Suivi CamShift", frame)
    cv2.imshow("Carte de couleur", dst)

    # Touche 'q' pour quitter
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()