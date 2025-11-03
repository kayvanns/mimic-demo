import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import OneHotEncoder
import ast

df = pd.read_csv("/Users/kayvans/Documents/mimic-demo/csv_files/uti_admission_wide_draft1.csv")
df = df.drop_duplicates(subset=['hadm_id'])
categorical_cols= ['gender', 'race', 'admission_type', 'admission_location']
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))
df =  pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

def parse(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["antibiotic_list"] = df["antibiotics"].apply(parse)
all_antibiotics = sorted({drug for lst in df["antibiotic_list"] for drug in lst})
for drug in all_antibiotics:
    df[f"{drug}_given"] = df["antibiotic_list"].apply(lambda lst: 1 if drug in lst else 0)
df["vasoactive_list"] = df["vasoactive_meds"].apply(parse)
all_vasoactives = sorted({drug for lst in df["vasoactive_list"] for drug in lst})
for drug in all_vasoactives:
    df[f"{drug}_given"] = df["vasoactive_list"].apply(lambda lst: 1 if drug in lst else 0)

df = df.drop(columns=["hadm_id","subject_id","stay_id","admittime","dischtime","end_window","start_window","intime","outtime","first_antibiotic_time",'antibiotics', 'vasoactive_meds', 'antibiotic_list', 'vasoactive_list'], axis=1)
print(df.head())
cols = [
    "temperature_max_F",
    "heart_rate_max",
    "Vancomycin_given",
"vent_or_intubation","Norepinephrine_given",
    "sepsis",
    "septic_shock","hospital_expire_flag"]

data = df[cols].to_numpy()
cg = pc(data, mvpc=True, indep_test='mv_fisherz')
cg.draw_pydot_graph(labels=cols)
pyd = GraphUtils.to_pydot(cg.G,labels=cols) 
pyd.write_png("mixed_mvpc_fisher.png")
cols = ["Vancomycin_given",
"vent_or_intubation","Norepinephrine_given",
    "sepsis",
    "septic_shock","hospital_expire_flag"]
data = df[cols].to_numpy()
cg = pc(data, indep_test='chisq')
cg.draw_pydot_graph(labels=cols)
pyd = GraphUtils.to_pydot(cg.G,labels=cols) 
pyd.write_png("categorical_chisq.png")

