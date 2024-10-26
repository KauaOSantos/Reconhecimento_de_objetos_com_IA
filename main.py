#Dependências
import time
import cv2

#Criando as cores das classes
COLORS = [(0,255,255),(255,255,0), (0,255,0), (255,0,0)]

#Criando as classes
class_names = []
with open("cocopt.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#Captura do vídeo
cap = cv2.VideoCapture("animal.mp4")
#cap = cv2.VideoCapture(0)

#Carregar os pesos da rede
#mais leve
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")
#mais completo
#net = cv2.dnn.readNet("yolov4.weights","yolov4.cfg")

#Cria um modelo a parir da rede criada - Crie um modelo a partir de uma rede de aprendizado profundo.

model = cv2.dnn_DetectionModel(net)

#model.setInputParams(size=(416,416),scale=1/225)
model.setInputParams(size=(416,416),scale=1/255)

#Lendo os frames do vídeo
while True:
    #Captura do frame
    _,frame = cap.read()
    
    #Contando o temspo ms
    start = time.time()
    
    #Detectando
    classes, scores, boxes = model.detect(frame, 0.1, 0.3) 

    #Encettand o time
    end = time.time()
    
    for (classid, score, box) in zip(classes, scores, boxes):
        #Gerando a cor da classe
        color = COLORS[int(classid) % len(COLORS)]
        #Pegando o nome da classe pelo ID e sua acurácia
        #label = f"{class_names[classid[0]]}:{score}"
        label = f"{class_names[classid]}:{score}"
        #Desenhando a caixa da detecção
        cv2.rectangle(frame, box, color, 2)
        #Escrevendo o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0],box[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    #Tempo para detecção
    fps_label = f"FPS: {round((1.0/(end-start)),2)}"
    
    #escrevendo fps na imagem
    #cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
    
    #Mostrando a imagem
    cv2.imshow("detections", frame)
    
    #Espera da resposta
    if cv2.waitKey(1) == 27:
        break
    
    #liberando a cam e destruindo a imagem
cap.release()
cv2.destroyAllWindows()