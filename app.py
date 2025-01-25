import gradio as gr
import pandas as pd
import joblib

# Load scaler, encoder, and model
scaler = joblib.load(r'toolkit\scaler.joblib')
model = joblib.load(r'toolkit\model_final.joblib')
encoder = joblib.load(r'toolkit\encoder.joblib')

# Prediction function
def predict(Gender, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI):
    # Create a DataFrame
    input_df = pd.DataFrame({
        'Gender': [Gender],
        'Urea': [float(Urea)],
        'Cr': [float(Cr)],
        'HbA1c': [float(HbA1c)],
        'Chol': [float(Chol)],
        'TG': [float(TG)],
        'HDL': [float(HDL)],
        'LDL': [float(LDL)],
        'VLDL': [float(VLDL)],
        'BMI': [float(BMI)],
    })

    # Transform gender using the encoder
    input_df['Gender'] = encoder.transform(input_df['Gender'])
    # Apply scaler if necessary (uncomment if scaling is required)
    # input_df = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df)

    # Prediction label mapping
    prediction_label = {0: "No Diabetes", 1: "Prediabetic", 2: "Diabetic"}
    return prediction_label[int(prediction[0])]

# Gradio app
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Image("diabete1.jpg", label="Diabetes Prediction App")
    gr.Markdown("# Diabetes Prediction App")
    gr.Markdown(
        "This app predicts a patient's diabetes status based on their health parameters. "
        "Please provide the following inputs:"
    )
    
    with gr.Row():
        with gr.Column():
            Gender = gr.Radio(['M', 'F'], label="Gender")
            Urea = gr.Slider(0, 100, step=1, label="Urea (mg/dL)")
            Cr = gr.Number(label="Creatinine (mg/dL)")
            HbA1c = gr.Number(label="HbA1c (%)")
            Chol = gr.Number(label="Cholesterol (mg/dL)")
        with gr.Column():
            TG = gr.Number(label="Triglycerides (mg/dL)")
            HDL = gr.Slider(0, 100, step=1, label="HDL Cholesterol (mg/dL)")
            LDL = gr.Number(label="LDL Cholesterol (mg/dL)")
            VLDL = gr.Number(label="VLDL (mg/dL)")
            BMI = gr.Number(label="Body Mass Index (BMI)")
    
    predict_btn = gr.Button("Predict")
    output = gr.Label(label="Prediction Result")

    predict_btn.click(
        fn=predict,
        inputs=[Gender, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI],
        outputs=output
    )

app.launch(share=True)
