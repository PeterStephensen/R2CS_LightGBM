# R → ONNX → C# LightGBM Pipeline

Komplet eksempel på at træne en binær LightGBM-model i R, konvertere til ONNX via Python og køre inference i C#.

## Mappestruktur

```
R2CS_LightGBM/
├── 01_train_lightgbm.R          # Trin 1: Træn model i R
├── 02_convert_to_onnx.py        # Trin 2: Konvertér til ONNX i Python
├── requirements.txt             # Python-afhængigheder
├── CSharp/OnnxInference/        # Trin 3: C#-inference
│   ├── OnnxInference.csproj
│   └── Program.cs
├── model/                       # Genereres af scripts
│   ├── lightgbm_model.txt       # LightGBM native format (fra R)
│   └── lightgbm_model.onnx      # ONNX-format (fra Python)
└── data/                        # Genereres af scripts
    ├── test_features.csv         # Testdata (4 features, 200 rækker)
    ├── test_predictions_r.csv    # R-predictions til verificering
    └── test_predictions_python.csv # Python/ONNX-predictions
```

## Forudsætninger

- **R** med `lightgbm`-pakken: `install.packages("lightgbm")`
- **Python 3.9+** med pakkerne i `requirements.txt`
- **.NET 8.0 SDK**

## Trin 1 – Træn LightGBM i R

```bash
Rscript 01_train_lightgbm.R
```

Genererer syntetisk binær klassifikationsdata (1000 samples, 4 features), træner en LightGBM-model og gemmer:

- `model/lightgbm_model.txt` – modelfilen
- `data/test_features.csv` – testdata (200 rækker)
- `data/test_predictions_r.csv` – R-sandsynligheder til sammenligning

## Trin 2 – Konvertér til ONNX i Python

```bash
pip install -r requirements.txt
python 02_convert_to_onnx.py
```

Indlæser LightGBM-modellen, konverterer til ONNX og verificerer at predictions matcher R-output. ZipMap-operatoren fjernes automatisk, så output er en ren float-tensor `[N, 2]` (nemmere at bruge i C#).

## Trin 3 – Kør inference i C#

```bash
cd CSharp/OnnxInference
dotnet run
```

Indlæser ONNX-modellen, kører inference på testdata og sammenligner med R/Python-predictions.

Alternativt med eksplicitte stier:

```bash
dotnet run -- "../../model/lightgbm_model.onnx" "../../data/test_features.csv"
```

## ONNX Model I/O

| Retning | Navn   | Type    | Shape     | Beskrivelse                          |
|---------|--------|---------|-----------|--------------------------------------|
| Input   | input  | float32 | [N, 4]   | 4 features: x1, x2, x3, x4         |
| Output  | label  | int64   | [N]      | Predicted class (0 eller 1)          |
| Output  | probs  | float32 | [N, 2]   | Sandsynligheder for class 0 og 1    |
