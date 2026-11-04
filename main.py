from utralytics import YOLO

#Carrega um modelo YOLO pré-treinado
model = YOLO("yolo26n.pt") 
#É usado para treinar um modelo YOLO em um conjunto de dados personalizado. O arquivo de configuração "coco8.yaml" contém informações sobre o conjunto de dados, como classes, caminhos para imagens e anotações, e outras configurações relevantes para o treinamento.
results = model.train(data="coco8.yaml",epochs=15)

# Avalia o desempenho do modelo no conjunto de validação
results = model.val()