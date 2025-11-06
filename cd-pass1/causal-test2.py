from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kayvans/Documents/mimic-demo/csv_files/uti_admission_wide_draft1.csv")
df = df.drop_duplicates(subset=['hadm_id'])
df = df.drop(columns=["hadm_id","subject_id","stay_id","admittime","dischtime","end_window","start_window","intime","outtime","first_antibiotic_time","uti","temperature_max_C"], axis=1)
categorical_cols= ['gender', 'race', 'admission_type', 'admission_location']

df["vasoactive_meds"] = df["vasoactive_meds"].notna().astype(int)
df["antibiotics"] = df["antibiotics"].notna().astype(int)

binary_cols = ['vent_or_intubation', 'sepsis', 'septic_shock', 'hospital_expire_flag','arf','antibiotics','vasoactive_meds']
cont_cols = df.select_dtypes(include=['float', 'int']).columns.difference(binary_cols)
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes
for col in binary_cols:
    df[col] = df[col].astype('int')
df[cont_cols] = SimpleImputer(strategy='median').fit_transform(df[cont_cols])
df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])
data = df.to_numpy()
cg = pc(data, indep_test='mv_fisherz', mvpc=True)
cg.draw_pydot_graph(labels=df.columns.tolist())