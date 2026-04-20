import cv2
import numpy as np
import mss

# Capture d'écran (il faut bien placer la vidéo sur l'écran de l'ordi car on lit l'écran de l'ordianteur)
sct = mss.mss()
zone_capture = {"top": 200, "left": 200, "width": 700, "height": 600} # valeurs de la fenêtre, à adapter si on change d'écran

print("Tracking en cours... (appuyez sur q pour arrêter)")

while True:
    # 1. Récupération de l'image
    capture = sct.grab(zone_capture)
    frame = np.array(capture)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # mss renvoie du BGRA, on passe en BGR pour OpenCV

    # 2. Passage en HSV et création du masque
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Seuils pour isoler le châssis noir du robot
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 45])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Nettoyage pour enlever les reflets de l'eau et le bruit
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Détection des contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # On suppose que le robot est le plus gros objet sombre à l'écran
        plus_grand_contour = max(contours, key=cv2.contourArea)
        aire = cv2.contourArea(plus_grand_contour)
        
        # Filtre pour ne pas tracker des bouts de câble ou des ombres (aire > 1000)
        if aire > 1000:
            x, y, w, h = cv2.boundingRect(plus_grand_contour)
            
            # Calcul du barycentre avec les moments spatiaux
            M = cv2.moments(plus_grand_contour)
            
            # Sécurité : on évite la division par zéro au cas où
            if M["m00"] != 0:
                bary_x = int(M["m10"] / M["m00"])
                bary_y = int(M["m01"] / M["m00"])
            else:
                # Si ça plante, on prend le milieu de la bounding box par défaut
                bary_x, bary_y = x + w // 2, y + h // 2 
            
            # Tracé de la boîte (vert) et du point central (rouge)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (bary_x, bary_y), 6, (0, 0, 255), -1) 
            
            # Affichage des coordonnées
            cv2.putText(frame, f"Barycentre: ({bary_x}, {bary_y})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "ROV perdu ou trop loin", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 4. Affichage pour le debug
    cv2.imshow("Vue Camera", frame)
    cv2.imshow("Masque HSV (Debug)", mask) # Pratique pour voir ce que l'algo isole vraiment

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()