"""
Indlæs LightGBM-model (gemt fra R) og konvertér til ONNX-format.
Fjerner ZipMap-operatoren så output er rene float-tensorer,
hvilket gør C#-inference væsentligt simplere.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import onnxmltools
import onnxruntime as rt
import onnx
from onnx import helper, TensorProto
from onnxconverter_common.data_types import FloatTensorType
from pathlib import Path


def remove_zipmap(onnx_model):
    """Fjern ZipMap-operator så sandsynligheder returneres som float-tensor [N,2]."""
    for node in list(onnx_model.graph.node):
        if node.op_type == "ZipMap":
            prob_input_name = node.input[0]
            zipmap_output_name = node.output[0]

            onnx_model.graph.node.remove(node)

            for i, output in enumerate(onnx_model.graph.output):
                if output.name == zipmap_output_name:
                    onnx_model.graph.output.remove(output)
                    new_output = helper.make_tensor_value_info(
                        prob_input_name, TensorProto.FLOAT, None
                    )
                    onnx_model.graph.output.insert(i, new_output)
                    break
            break

    return onnx_model


def main():
    model_txt = Path("model/lightgbm_model.txt")
    model_onnx = Path("model/lightgbm_model.onnx")
    test_csv = Path("data/test_features.csv")

    booster = lgb.Booster(model_file=str(model_txt))
    num_features = booster.num_feature()
    print(f"Indlæst LightGBM-model med {num_features} features")

    initial_types = [("input", FloatTensorType([None, num_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        booster, initial_types=initial_types, target_opset=15
    )

    onnx_model = remove_zipmap(onnx_model)
    onnx.save_model(onnx_model, str(model_onnx))
    print(f"ONNX-model gemt: {model_onnx}")

    sess = rt.InferenceSession(str(model_onnx))

    print("\nONNX model-info:")
    for inp in sess.get_inputs():
        print(f"  Input:  {inp.name}  shape={inp.shape}  type={inp.type}")
    for out in sess.get_outputs():
        print(f"  Output: {out.name}  shape={out.shape}  type={out.type}")

    test_df = pd.read_csv(test_csv)
    X_test = test_df.values.astype(np.float32)
    print(f"\nTestdata: {X_test.shape[0]} rækker, {X_test.shape[1]} kolonner")

    input_name = sess.get_inputs()[0].name
    results = sess.run(None, {input_name: X_test})

    labels = results[0]
    probs = results[1]

    print(f"Predictions shape: labels={labels.shape}, probs={probs.shape}")
    print(f"\nFørste 5 ONNX-predictions (P(class=1)):")
    for i in range(min(5, len(probs))):
        print(f"  Sample {i}: {probs[i, 1]:.6f}")

    pd.DataFrame({"probability": probs[:, 1]}).to_csv(
        "data/test_predictions_python.csv", index=False
    )

    r_preds = pd.read_csv("data/test_predictions_r.csv")
    max_diff = np.max(np.abs(r_preds["probability"].values - probs[:, 1]))
    mean_diff = np.mean(np.abs(r_preds["probability"].values - probs[:, 1]))
    print(f"\nSammenligning R vs. ONNX:")
    print(f"  Max  afvigelse: {max_diff:.10f}")
    print(f"  Mean afvigelse: {mean_diff:.10f}")


if __name__ == "__main__":
    main()
