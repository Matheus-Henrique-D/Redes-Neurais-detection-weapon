# weapon_detection_v4

Dataset de detecção de pessoas armadas em imagens de câmeras de segurança, no formato YOLO (Ultralytics).
Snapshot de 2026-10-16: 1545 imagens rotuladas e revisadas.

## Classes
| id | nome          | instâncias |
|----|---------------|-----------:|
| 0  | pessoa_armada |        763 |
| 1  | vitima        |        754 |
| 2  | pessoa        |       3673 |

## Estrutura
```
data.yaml
images/{train,val,test}/   imagens
labels/{train,val,test}/   um .txt por imagem: "classe x_centro y_centro largura altura" (normalizados 0–1)
split.csv                  origem de cada imagem e seu grupo (vídeo)
```

## Divisão
| split | imagens |
|-------|--------:|
| train |    1133 |
| val   |     208 |
| test  |     204 |

A divisão é **agrupada por vídeo**: todos os frames de um mesmo vídeo ficam no mesmo split
(coluna `group` do `split.csv`). Assim não há frames quase idênticos vazando entre treino e teste.
Use `val` para escolher hiperparâmetros e deixe `test` só para a avaliação final.

## Treino rápido
```bash
pip install ultralytics
yolo detect train data=data.yaml model=yolo26n.pt imgsz=640 epochs=100
yolo detect val data=data.yaml model=runs/detect/train/weights/best.pt split=test
```
Rode os comandos de dentro desta pasta (o `data.yaml` usa `path: .`).
