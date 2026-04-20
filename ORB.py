import cv2
import numpy as np
import os

# Initialisation de l'extracteur ORB (2000 points max)
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Chargement des images de référence (modèles CAO du robot)
path_refs = 'imagesCAO2' 
references = []

print(f"Chargement des références depuis '{path_refs}'...")
for file in os.listdir(path_refs):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(path_refs, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                # On stocke les points, descripteurs, dimensions et l'image
                references.append((kp, des, img.shape, file, img))
                print(f" -> {file} chargé ({len(kp)} points)")

if not references:
    print("Erreur : Aucune référence trouvée.")
    exit()

# Lecture de la vidéo de test
cap = cv2.VideoCapture('BillyBassinX2.mov')

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    
    best_match_count = 0
    best_good_matches = []
    best_kp_ref, best_shape_ref, best_img_ref = None, None, None
    best_name = ""
    
    if des_frame is not None:
        # Comparaison avec toutes les images de référence
        for kp_ref, des_ref, shape_ref, name, img_ref in references:
            matches = bf.knnMatch(des_ref, des_frame, k=2)
            
            # Filtre de Lowe pour écarter les fausses correspondances
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Conservation de la référence avec le plus de correspondances
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_good_matches = good_matches
                best_kp_ref = kp_ref
                best_shape_ref = shape_ref
                best_name = name
                best_img_ref = img_ref 
        
        # Validation de la cible (seuil fixé à 15 points minimum)
        if best_match_count > 15:
            src_pts = np.float32([best_kp_ref[m.queryIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
            
            # Calcul de l'homographie avec RANSAC pour éliminer les aberrations (outliers)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Dessin du contour projeté du robot sur la vidéo
                h, w = best_shape_ref
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"ROV Detecte ({best_name})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Affichage des lignes de correspondance (uniquement les inliers validés par RANSAC)
            matches_mask = mask.ravel().tolist() if mask is not None else None
            img_matches = cv2.drawMatches(
                best_img_ref, best_kp_ref, 
                frame, kp_frame, 
                best_good_matches, None, 
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matches_mask, 
                flags=2 
            )
            
            cv2.namedWindow("Correspondances ORB", cv2.WINDOW_NORMAL)
            cv2.imshow("Correspondances ORB", img_matches)

        else:
            cv2.putText(frame, "Recherche en cours...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
            
    cv2.imshow('Detection ORB', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()