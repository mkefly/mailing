import streamlit as st
import pandas as pd
from livedesk_mailing_lists.inference.inference import run_inference

def main():
    st.title("Inference Pipeline")

    X_infer = pd.DataFrame({
        'feature1': [0.1, 0.4, 0.6],
        'feature2': [0.2, 0.5, 0.7],
    })

    model_name = st.text_input("Model Name", "your_model_name_here")
    model_version = st.number_input("Model Version", 1, step=1)
    thresholds = st.multiselect("Thresholds", options=[0.3, 0.5, 0.7], default=[0.5])
    z_threshold = st.slider("Z-score Threshold", 0.0, 3.0, 1.0)

    if st.button("Run Inference"):
        try:
            results = run_inference(model_name, model_version, X_infer, thresholds, z_threshold)
            st.write("Threshold Results", results['threshold_results'])
            st.write("Impact Analysis Results", results['impact_analysis_results'])
        except RuntimeError as e:
            st.error(f"Inference workflow failed: {str(e)}")

if __name__ == "__main__":
    main()
