"""Treina e testa um detector YOLO pré-treinado para o dataset do projeto."""

import os
from glob import glob
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
PASTA_DADOS = ROOT_DIR / "weapon_detection_v4" / "weapon_detection_v4"
DADOS = PASTA_DADOS / "data.yaml"
MODELO_BASE = "yolo26n.pt"
EPOCAS = 100
TAMANHO_IMG = 640


def treinar() -> None:
    # O data.yaml usa caminhos relativos à pasta do dataset.
    os.chdir(PASTA_DADOS)

    # Começa com pesos pré-treinados e ajusta o modelo às classes do projeto.
    modelo = YOLO(MODELO_BASE)
    modelo.train(
        data=DADOS.name,
        epochs=EPOCAS,
        patience=20,
        imgsz=TAMANHO_IMG,
        batch=16,
        project=str(ROOT_DIR / "runs" / "detect"),
        name="weapon_detection_yolo",
    )

    # Usa os melhores pesos encontrados durante o treino.
    melhores_pesos = modelo.trainer.best
    modelo = YOLO(str(melhores_pesos))

    # Avalia no conjunto de validação; o conjunto de teste fica reservado.
    modelo.val(
        data=DADOS.name,
        split="val",
        project=str(ROOT_DIR / "runs" / "detect"),
        name="val_yolo",
    )

    # Mostra uma predição em uma imagem de validação.
    imagens = sorted(glob(str(PASTA_DADOS / "images" / "val" / "*.png")))
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem PNG encontrada em {PASTA_DADOS / 'images' / 'val'}")

    modelo.predict(
        source=imagens[0],
        conf=0.25,
        save=True,
        project=str(ROOT_DIR / "runs" / "detect"),
        name="predict_yolo",
    )


if __name__ == "__main__":
    treinar()
