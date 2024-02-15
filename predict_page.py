import pickle
import numpy as np
import pandas as pd
import streamlit as st


def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
sample = data["sample"]
new_attrs = ['grow_policy', 'max_bin', 'eval_metric', 'callbacks', 'early_stopping_rounds', 'max_cat_to_onehot', 'max_leaves', 'sampling_method']

for attr in new_attrs:
    setattr(model, attr, None)
def show_predict_page():
    st.title("Textbook Outcome Following CRS-HIPEC")

    st.info("###### 📚 This risk calculator was developed and validated using data from a cohort of 1,954 adult patients who underwent CRS-HIPEC between 2001 and 2020. (AUC: 0.76)")
    st.info("###### 💡 Tip: The variables have set ranges determined by our dataset. Should there be a need to enter a value beyond these ranges, please use the maximum or minimum value available.")

    st.write("""### Please  provide the following information:""")
    
    Age = st.slider("Age", 18, 90, 18)
    AlbumingdL = st.slider("Albumin", 0, 10, 3)
    PCI = st.slider("Peritoneal cancer index (PCI)", 0, 39, 10)
    CCI = st.slider("Charlson comorbidity index (CCI)", 0, 24, 0)
    Symptomatic =  st.selectbox('Symptomatic', ("No", "Yes"))
    proctectomy =  st.selectbox('Proctectomy', ("No", "Yes"))
    PartialColectomy =  st.selectbox('Partial Colectomy', ("No", "Yes"))
    Splenectomy =  st.selectbox('Splenectomy', ("No", "Yes"))




    ok = st.button("Predict the chance of textbook outcome")

    if ok:
        sample["Age"] = Age

        sample["AlbumingdL"] = AlbumingdL
        sample["PCI"] = PCI
        sample["CCI"] = CCI
        
        if  PartialColectomy == "Yes":
            sample["PartialColectomy"] = 1
        if  Splenectomy == "Yes":
            sample["Splenectomy"] = 1
        if  proctectomy == "Yes":
            sample["proctectomy"] = 1
        if  Symptomatic == "Yes":
            sample["Symptomatic"] = 1




        chance = model.predict_proba(sample)
        #sample = data["sample"]
        st.subheader(f"Estimated chance of textbook outcome: {chance[0][1]*100:.2f}%")
    reset = st.button("Reset")
    if reset:
        sample.loc[:,:] = 0

    st.error("###### ❗ Disclaimer: Please note that this tool does not reflect causal relationships between input variables and the outcome, and therefore it should not be used in isolation to dictate surgical planning.")






        

