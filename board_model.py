"""
export_model_to_arduino.py
Writes two files:
  model_params.h   — declarations only (extern)
  model_params.cpp — actual data definitions (defined once, linked once)

This is the correct pattern for ESP32/Arduino. Defining large const arrays
directly in a .h file causes duplicate symbol bugs when the header is included
from more than one translation unit — the linker silently picks one copy which
may be zero-initialized, producing garbage scores and conf=1.000.
"""

import joblib
import numpy as np

model  = joblib.load("svm_retrain_model.pkl")
scaler = joblib.load("svm_retrain_scaler.pkl")

sv        = model.support_vectors_
dc        = model.dual_coef_
intercept = model.intercept_
gamma     = model._gamma
probA     = model.probA_
probB     = model.probB_
classes   = model.classes_
n_support = model.n_support_

n_sv, n_feat = sv.shape
n_classes    = len(classes)
n_pairs      = n_classes * (n_classes - 1) // 2
sc_mean      = scaler.mean_
sc_scale     = scaler.scale_
sv_start     = np.concatenate([[0], np.cumsum(n_support[:-1])])

print(f"Classes   : {list(classes)}")
print(f"n_sv      : {n_sv}   n_feat : {n_feat}")
print(f"n_classes : {n_classes}   n_pairs : {n_pairs}")
print(f"SVs/class : {n_support}")
print(f"gamma     : {gamma}")

# ── Pre-flatten OvO dual coefs ────────────────────────────────────────────────
pair_coef = np.zeros((n_pairs, n_sv), dtype=np.float64)
pair_i_list, pair_j_list = [], []
pair_idx = 0
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        si = slice(sv_start[i], sv_start[i] + n_support[i])
        sj = slice(sv_start[j], sv_start[j] + n_support[j])
        pair_coef[pair_idx, si] = dc[j - 1, si]
        pair_coef[pair_idx, sj] = dc[i,     sj]
        pair_i_list.append(i)
        pair_j_list.append(j)
        pair_idx += 1

# ── Helpers ───────────────────────────────────────────────────────────────────
def c_float_array(name, data):
    flat = np.array(data, dtype=np.float64).flatten()
    vals = ",\n    ".join(f"{v:.10f}f" for v in flat)
    n = len(flat)
    return f"const float {name}[{n}] = {{\n    {vals}\n}};\n"

def c_uint8_array(name, data):
    vals = ", ".join(str(int(x)) for x in data)
    return f"const uint8_t {name}[{len(data)}] = {{{vals}}};\n"

# ── Write model_params.h (declarations only) ──────────────────────────────────
h_lines = []
h_lines.append("// ================================================================")
h_lines.append("// model_params.h  —  AUTO-GENERATED, DO NOT EDIT")
h_lines.append("// Declarations only. Data lives in model_params.cpp.")
h_lines.append("// This pattern avoids duplicate-symbol bugs on ESP32/Arduino.")
h_lines.append("// ================================================================")
h_lines.append("#pragma once")
h_lines.append("#include <stdint.h>")
h_lines.append("")
h_lines.append(f"#define N_SUPPORT   {n_sv}")
h_lines.append(f"#define N_FEATURES  {n_feat}")
h_lines.append(f"#define N_CLASSES   {n_classes}")
h_lines.append(f"#define N_PAIRS     {n_pairs}")
h_lines.append(f"#define GAMMA       {gamma:.10f}f")
h_lines.append("")

label_str = ", ".join(f'"{c}"' for c in classes)
h_lines.append(f'extern const char* const CLASS_NAMES[N_CLASSES];')
h_lines.append("")
h_lines.append("extern const float   SC_MEAN[N_FEATURES];")
h_lines.append("extern const float   SC_SCALE[N_FEATURES];")
h_lines.append(f"extern const float   SUPPORT_VECTORS[N_SUPPORT * N_FEATURES];")
h_lines.append(f"extern const float   PAIR_COEF[N_PAIRS * N_SUPPORT];")
h_lines.append("extern const float   INTERCEPT[N_PAIRS];")
h_lines.append("extern const float   PROB_A[N_PAIRS];")
h_lines.append("extern const float   PROB_B[N_PAIRS];")
h_lines.append("extern const uint8_t PAIR_I[N_PAIRS];")
h_lines.append("extern const uint8_t PAIR_J[N_PAIRS];")

with open("model_params.h", "w") as f:
    f.write("\n".join(h_lines) + "\n")
print("✅  Wrote model_params.h")

# ── Write model_params.cpp (data definitions — compiled once) ─────────────────
cpp_lines = []
cpp_lines.append("// ================================================================")
cpp_lines.append("// model_params.cpp  —  AUTO-GENERATED, DO NOT EDIT")
cpp_lines.append("// ================================================================")
cpp_lines.append('#include "model_params.h"')
cpp_lines.append("")

label_str = ", ".join(f'"{c}"' for c in classes)
cpp_lines.append(f'const char* const CLASS_NAMES[N_CLASSES] = {{{label_str}}};')
cpp_lines.append("")
cpp_lines.append(c_float_array("SC_MEAN",          sc_mean))
cpp_lines.append(c_float_array("SC_SCALE",         sc_scale))
cpp_lines.append(c_float_array("SUPPORT_VECTORS",  sv))
cpp_lines.append(c_float_array("PAIR_COEF",        pair_coef))
cpp_lines.append(c_float_array("INTERCEPT",        intercept))
cpp_lines.append(c_float_array("PROB_A",           probA))
cpp_lines.append(c_float_array("PROB_B",           probB))
cpp_lines.append(c_uint8_array("PAIR_I",           pair_i_list))
cpp_lines.append(c_uint8_array("PAIR_J",           pair_j_list))

with open("model_params.cpp", "w") as f:
    f.write("\n".join(cpp_lines) + "\n")
print("✅  Wrote model_params.cpp")

# ── Sanity check values ───────────────────────────────────────────────────────
print("\n=== SANITY CHECK (compare with Arduino serial) ===")
print(f"SC_MEAN[0..2]:      {sc_mean[0]:.6f}  {sc_mean[1]:.6f}  {sc_mean[2]:.6f}")
print(f"SC_SCALE[0..2]:     {sc_scale[0]:.6f}  {sc_scale[1]:.6f}  {sc_scale[2]:.6f}")
print(f"SV[0][0..2]:        {sv[0,0]:.6f}  {sv[0,1]:.6f}  {sv[0,2]:.6f}")
print(f"PAIR_COEF[0][0..2]: {pair_coef[0,0]:.6f}  {pair_coef[0,1]:.6f}  {pair_coef[0,2]:.6f}")
print(f"INTERCEPT[0..2]:    {intercept[0]:.6f}  {intercept[1]:.6f}  {intercept[2]:.6f}")
print("==================================================")
print("\nCopy model_params.h AND model_params.cpp into your sketch folder.")
print("Both files must be in the same folder as gesture_classifier.ino")