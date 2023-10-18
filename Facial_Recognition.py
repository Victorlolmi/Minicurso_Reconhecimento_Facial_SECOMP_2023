
# Reconhecimento Facial com DeepFace

#1.1 Carregar o video

import cv2

# Carregue o vídeo do disco
video_filename = "vids\leon&nilce_curto.mp4"
video_capture_input = cv2.VideoCapture(video_filename)

#1.2 Criar video de saida

import imageio

# Defina o arquivo de vídeo de saída
fps = int(video_capture_input.get(cv2.CAP_PROP_FPS))
video_capture_output = imageio.get_writer("vids\output.mp4", fps = fps)

#1.3 De os caminhos para as imagens de referencia

# Imagem do Leon
target_image_leon = "imgs\img_Leon.jpg"
# Imagem da Nilce
target_image_nilce= "imgs\img_Nilce.jpg"

#Utilizacao do DeepFace

#2.1 Gerar reconhecimento facial para todos os frames 


from deepface import DeepFace

frame_counter = 0

while True:
    # Para cada frame
    success, frame = video_capture_input.read()


    frame_counter += 1

    print("Frame: "+frame_counter)
    # Se o vídeo acabou, saia do loop
    if not success:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    '''
    # "Pula" alguns frames
    if frame_counter % 2 != 0:
        continue
    '''
   
    
    # Detecte as faces do leon no frame 
    detected_face_leon = DeepFace.verify(frame, target_image_leon, "VGG-Face",enforce_detection=False)
    #dimencoes do frame do Leon
    (x_leon, y_leon, w_leon, h_leon) = (int(detected_face_leon['facial_areas']['img1']['x']), int(detected_face_leon['facial_areas']['img1']['y']), int(detected_face_leon['facial_areas']['img1']['w']), int(detected_face_leon['facial_areas']['img1']['h']))


    # Detecte as faces da Nilce no frame 
    detected_face_nilce = DeepFace.verify(frame, target_image_nilce, "VGG-Face",enforce_detection=False)
    #dimencoes do frame da Nilce
    (x_nilce, y_nilce, w_nilce, h_nilce) = (int(detected_face_nilce['facial_areas']['img1']['x']), int(detected_face_nilce['facial_areas']['img1']['y']), int(detected_face_nilce['facial_areas']['img1']['w']), int(detected_face_nilce['facial_areas']['img1']['h']))

    if detected_face_leon['verified'] == True and detected_face_leon['similarity_metric'] == "cosine":

        recognized_name = "Leon"
        print("Leon reconhecido")

        cv2.rectangle(frame, (x_leon, y_leon), (x_leon + w_leon, y_leon + h_leon), (0, 255, 0), 4)
        cv2.putText(frame, recognized_name, (x_leon, y_leon - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
    if detected_face_nilce['verified'] == True and detected_face_nilce['similarity_metric'] == "cosine":

        recognized_name = "Nilce"

        if h_nilce >= 150 and h_nilce <=  250:

            print("Nilce reconhecida")

            cv2.rectangle(frame, (x_nilce, y_nilce), (x_nilce + w_nilce, y_nilce + h_nilce), (0, 255, 0), 4)
            cv2.putText(frame, recognized_name, (x_nilce, y_nilce - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
    # Escreva o frame no arquivo de vídeo de saída
    
    video_capture_output.append_data(frame)

#2.2 Libere os objetos de captura e gravacao de video

video_capture_input.release()
video_capture_output.close()