import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import pickle
from tqdm import tqdm
import imageio

def generateFolderWithFaces(output_folder, video_path, target_image_path, threshold_faces=2, step_frame=50):
    detected_faces_list = []  # Lista para armazenar as faces detectadas
    detected_faces_counter = 0
    frame_counter = 0

    # Carregue a imagem alvo
    target_image = cv2.imread(target_image_path)

    # Carregue o vídeo
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Contabiliza frames
        frame_counter += 1

        # Pega o próximo frame
        ret, frame = cap.read()

        # Acabou o vídeo ou detectou o limite de faces?
        if not ret or detected_faces_counter >= threshold_faces:
            break

        # "Pula" alguns frames
        if frame_counter % step_frame != 0:
            continue

        # Detecte as faces no frame usando DeepFace.detectFace
        detected_faces = DeepFace.verify(frame, target_image, "VGG-Face", enforce_detection=False)
        

        if detected_faces ['verified'] == True and detected_faces['similarity_metric'] == "cosine":
            # Total de faces detectadas até o momento
            detected_faces_counter += 1
            for i, face in enumerate(detected_faces):
                # Verifique se a face é igual à imagem alvo
                (x, y, w, h) = (int(detected_faces['facial_areas']['img1']['x']), int(detected_faces['facial_areas']['img1']['y']), int(detected_faces['facial_areas']['img1']['w']), int(detected_faces['facial_areas']['img1']['h']))
                face_crop = frame[y:y + h, x:x + w]

                print(face_crop)

                # Adicionar a face detectada à lista
                detected_faces_list.append(face_crop)

                # Salvar região das faces em uma pasta
                img_path = os.path.join(output_folder, f"face_{detected_faces_counter}_{frame_counter}.jpg")
                cv2.imwrite(img_path, face_crop)


                if h >= 150 and h <=  280:
                    # Adicionar retângulo ao redor da face e marcações para exibir bonitinho
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)

                 
    

    # Fechar o vídeo após processamento
    cap.release()
    
    # Retornar a lista de faces detectadas
    return detected_faces_list



# Caminho para o vídeo de entrada
video_path = 'ReconhecimentoFacial/download/vid_leon.mp4'

# Caminho para a foto do rosto da pessoa desejada
foto_path = 'ReconhecimentoFacial/Past_Leon/img1.jpg'

# Pasta para salvar as imagens recortadas
folder_name = 'ReconhecimentoFacial/Past_Leon'

#Defina um limiar de correspondência (ajuste conforme necessário)


Leon_faces = generateFolderWithFaces(folder_name, video_path, foto_path)

# Caminho para o vídeo de entrada
video_path = 'ReconhecimentoFacial/download/leon&nilce_completo.mp4'

# Caminho para a foto do rosto da pessoa desejada
foto_path = 'ReconhecimentoFacial/Past_Nilce/img1.jpg'

# Pasta para salvar as imagens recortadas
folder_name = 'ReconhecimentoFacial\Past_Nilce'

#Nilce_faces = generateFolderWithFaces(folder_name, video_path, foto_path)


# len Retona o numero de objetos num vetor 
num_faces = len(Leon_faces)
print(num_faces)

# Cria um subplot com linhas, Colunas e Dimencções da forma desejada

fig, axes = plt.subplots(nrows = 1, ncols = int(num_faces/8), figsize=(16, 16))
i = 0

for x in range(0, 2):
    axes[x].set_title('Face ' + str(i/8), fontsize = 14)
    Leon_faces[i] = cv2.cvtColor(Leon_faces[i], cv2.COLOR_BGR2RGB)
    axes[x].imshow(Leon_faces[i])
    i += 8

plt.show()