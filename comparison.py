import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from explainability.lime_explainer import explain_with_lime
from explainability.shap_explainer import explain_with_shap
from explainability.gradcam import apply_gradcam


def comparison_tab(load_image_model):

    st.header("📊 Comparaison des méthodes XAI (Image)")

    st.info(
        "Cette section compare plusieurs méthodes d’explicabilité appliquées "
        "au même modèle et à la même image afin d’analyser leurs complémentarités."
    )

    # --------------------------------------------------
    # INPUTS
    # --------------------------------------------------
    file = st.file_uploader(
        "Uploader un CT-scan",
        type=["png", "jpg", "jpeg"],
        key="compare_img"
    )

    arch = st.radio(
        "Architecture",
        ["AlexNet", "DenseNet121"],
        key="compare_arch"
    )

    # --------------------------------------------------
    # RUN COMPARISON
    # --------------------------------------------------
    if file and st.button("🚀 Lancer la comparaison"):

        # ---------- Load & preprocess ----------
        img_raw = Image.open(file).convert("RGB")
        img_np = np.array(img_raw.resize((224, 224)))

        tr = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        img_t = tr(img_raw).unsqueeze(0)
        model_img = load_image_model(arch)

        # ---------- Prediction ----------
        with torch.no_grad():
            p = torch.sigmoid(model_img(img_t)).item()

        label = "CANCER" if p > 0.5 else "SAIN"
        conf = (p if p > 0.5 else 1 - p) * 100

        st.metric("Diagnostic", label, f"{conf:.2f}%")

        # --------------------------------------------------
        # LEVEL 1 — CONFIDENCE
        # --------------------------------------------------
        st.markdown("### 🧠 Level 1 — Confidence")
        st.markdown(
            "Le score de confiance indique le degré de certitude du modèle, "
            "mais ne fournit aucune information sur les raisons de la décision."
        )

        # --------------------------------------------------
        # LEVEL 2 & 3 — XAI METHODS
        # --------------------------------------------------
        lime_img, lime_scores = explain_with_lime(
            model_img,
            img_np,
            is_torch=True
        )

        background = np.stack([
            img_np + np.random.normal(0, 5, img_np.shape)
            for _ in range(5)
        ]).clip(0, 255)

        shap_img = explain_with_shap(
            model_img,
            img_np,
            background
        )

        gradcam_img = apply_gradcam(
            model_img,
            img_t,
            img_np
        )

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.image(img_raw, caption="CT original", use_container_width=True)
        c2.image(lime_img, caption="LIME (Level 3 — local)", use_container_width=True)
        c3.image(shap_img, caption="SHAP (Level 3 — global)", use_container_width=True)
        c4.image(gradcam_img, caption="Grad-CAM (Level 2 — attention)", use_container_width=True)

        # --------------------------------------------------
        # INTERPRETATION / COMPARISON
        # --------------------------------------------------
        st.markdown("### 🧪 Analyse comparative")

        st.markdown(
            "- **Grad-CAM** met en évidence les régions activant le plus fortement le réseau "
            "(attention spatiale).\n"
            "- **LIME** fournit une explication locale en identifiant les superpixels ayant "
            "le plus d’impact sur la prédiction.\n"
            "- **SHAP** propose une vision plus globale et théorique de l’importance des régions."
        )

        st.success(
            "Lorsque plusieurs méthodes mettent en évidence des zones similaires, "
            "la confiance dans l’interprétation augmente. "
            "Des divergences indiquent une décision plus diffuse ou complexe."
        )

        # --------------------------------------------------
        # OPTIONAL: SHOW TOP LIME REGIONS (TEXTUAL)
        # --------------------------------------------------
        st.markdown("### 📊 Régions les plus influentes (LIME)")
        for zone, weight in lime_scores:
            st.caption(f"Zone {zone} — Impact relatif : {abs(weight):.4f}")
            st.progress(min(abs(weight) * 5, 1.0))
