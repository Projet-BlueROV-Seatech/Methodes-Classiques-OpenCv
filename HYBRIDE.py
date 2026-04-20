import cv2
import numpy as np
import os
import mss
import time

# Initialisation d'ORB pour la reconnaissance de forme
orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Chargement des images de référence (CAO)
path_refs = "imagesCAO2"
references = []

print("Chargement des modèles ORB...")
for file in os.listdir(path_refs):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(path_refs, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                references.append((kp, des, img.shape, file))

if not references:
    raise ValueError("Aucune image de référence chargée.")

# Zone de l'écran à filmer
capture_zone = {"top": 200, "left": 0, "width": 800, "height": 610}

# Plage de couleurs pour isoler le châssis sombre (HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 45])
kernel_morpho = np.ones((5, 5), np.uint8)

# Variables d'état du programme
tracking_active = False
tracker = None
current_target_name = ""
last_orb_time = time.time()

# Mémoire de position pour éviter que la détection saute sur une ombre lointaine
last_known_center = None 
MAX_DISTANCE = 250 

print("Démarrage du système... Appuyez sur 'q' pour quitter.")

with mss.mss() as sct:
    while True:
        # Capture de la frame
        sct_img = sct.grab(capture_zone)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- PHASE 1 : RECHERCHE (HSV + ORB) ---
        if not tracking_active:
            
            # 1. Filtre de couleur pour trouver les zones sombres
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_black, upper_black)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morpho)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morpho)

            mask_debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidat_valide = None

            if contours:
                # On trie les contours du plus grand au plus petit
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for c in contours:
                    aire = cv2.contourArea(c)
                    if aire < 1000:
                        break # Ignorer les parasites

                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Calcul du barycentre du candidat
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        bary_x = int(M["m10"] / M["m00"])
                        bary_y = int(M["m01"] / M["m00"])
                    else:
                        bary_x, bary_y = x + w // 2, y + h // 2

                    # Si le candidat est trop loin de l'ancienne position, on l'ignore (anti-saut)
                    if last_known_center is not None:
                        dist = np.sqrt((bary_x - last_known_center[0])**2 + (bary_y - last_known_center[1])**2)
                        if dist > MAX_DISTANCE:
                            continue 
                    
                    candidat_valide = (x, y, w, h, bary_x, bary_y)
                    break 

            if candidat_valide:
                x, y, w, h, bary_x, bary_y = candidat_valide
                
                cv2.rectangle(mask_debug, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.circle(mask_debug, (bary_x, bary_y), 5, (0, 0, 255), -1)   
                cv2.putText(mask_debug, "Candidat Evalue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # 2. Validation du candidat avec ORB
                roi_gray = gray_frame[y:y+h, x:x+w]
                kp_v, des_v = orb.detectAndCompute(roi_gray, None)

                max_good_matches = 0

                if des_v is not None and len(des_v) > 2:
                    for kp_r, des_r, shape_r, name in references:
                        matches = bf.knnMatch(des_r, des_v, k=2)
                        good = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.85 * m_n[1].distance]
                        
                        if len(good) > max_good_matches:
                            max_good_matches = len(good)
                            current_target_name = name

                # 3. Lancement du suivi continu si validation réussie
                if max_good_matches > 100:
                    tracker = cv2.TrackerMIL_create()
                    tracker.init(frame, (x, y, w, h))
                    tracking_active = True
                    last_orb_time = time.time()
                    
                    last_known_center = (bary_x, bary_y)
                    print(f"[{time.strftime('%H:%M:%S')}] Cible validée ({max_good_matches} pts). Démarrage MIL.")

            cv2.imshow("Vue Ordinateur (Filtre HSV)", mask_debug)
            cv2.putText(frame, "Recherche en cours...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Tracking BlueROV', frame) 

        # --- PHASE 2 : SUIVI FLUIDE (MIL) ---
        else:
            success, bbox = tracker.update(frame)

            if success:
                tx, ty, tw, th = [int(v) for v in bbox]
                
                bary_x = int(tx + tw/2)
                bary_y = int(ty + th/2)
                last_known_center = (bary_x, bary_y) 
                
                cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255, 255, 0), 2)
                cv2.circle(frame, (bary_x, bary_y), 5, (0, 0, 255), -1) 
                
                cv2.putText(frame, f"Barycentre: ({bary_x}, {bary_y})", (tx, ty - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Suivi MIL: {current_target_name}", (tx, ty - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Relance discrète d'une validation ORB toutes les 2 secondes
                if time.time() - last_orb_time >= 2.0:
                    tracking_active = False
            else:
                # Perte de la cible, on repart de zéro
                tracking_active = False
                last_known_center = None 
                print("Cible perdue. Réinitialisation complète.")

            # Masquage de la vue debug pendant le suivi
            masque_vide = np.zeros_like(frame)
            cv2.putText(masque_vide, "Tracking MIL actif", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Vue Ordinateur (Filtre HSV)", masque_vide)
            
            cv2.imshow('Tracking BlueROV', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cv2.destroyAllWindows()