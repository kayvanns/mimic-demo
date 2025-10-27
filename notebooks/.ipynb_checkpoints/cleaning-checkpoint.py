## Flow == get admissions with associated patients and icu stays based on diagnoses title + datetime conversions, calculation of stays --> get vitals by creating start and end window then collects highest/lowest measurements within x hours of ICU admission from chartevents based on vitals dict --> get max creatinine and bun from labevents --> get antibiotics and vasoactive agents taken within x hours of ICU admission from pharmacy

# 1. clean():
#       - Extract patients with diagnosis title
#       - Merge with admissions + ICU stays
#       - Calculate hospital and ICU stay duration
#       - Pass to get_vitals()
# 2. get_vitals():
#       - Create time window around ICU admission: intime - before to intime + after
#       - Pull vitals from chartevents within this window using vitals dict (itemid + agg function)
#       - Pass to get_max_creatinine_bun()
# 3. get_max_creatinine_bun():
#       - Pull max creatinine and BUN during hospital stay
#       - Pass to get_medications()
# 4. get_medications():
#       - Find antibiotics + vasopressor usage in same time window
#       - Pass to procedures

import pandas as pd
from datetime import timedelta
import datetime as dt

## reading files and creating dataframes for tables

path = "../data/mimic-iv-clinical-database-demo-2.2/hosp/"
path2 = "../data/mimic-iv-clinical-database-demo-2.2/icu/"

files = ["admissions", "patients", "labevents", "d_labitems", "prescriptions","pharmacy","transfers", "diagnoses_icd","d_icd_diagnoses","procedures_icd","d_icd_procedures"]
dfs = {}

for name in files:
    dfs[name] = pd.read_csv(path + f"{name}.csv.gz")

admissions = dfs["admissions"]
patients = dfs["patients"]
labs = dfs["labevents"]
d_labitems = dfs["d_labitems"]
prescriptions = dfs["prescriptions"]
pharmacy = dfs["pharmacy"]
transfers=dfs["transfers"]
diagnoses = dfs["diagnoses_icd"]
d_diagnoses = dfs["d_icd_diagnoses"]
procedures = dfs["procedures_icd"]
d_procedures = dfs["d_icd_procedures"]

files2 = ["icustays", "inputevents", "outputevents", "procedureevents","chartevents", "datetimeevents", "d_items"]
dfs2 = {}

for name in files2:
    dfs2[name] = pd.read_csv(path2 + f"{name}.csv.gz")
icustays = dfs2["icustays"]
inputevents = dfs2["inputevents"]
outputevents = dfs2["outputevents"]
procedureevents = dfs2["procedureevents"]
chartevents = dfs2["chartevents"]
datetimeevents = dfs2["datetimeevents"]
d_items = dfs2["d_items"]

columns = ['hadm_id','subject_id','stay_id',
    'anchor_age',
    'gender',
    'race',
    'admission_type',
    'admission_location',
    'admittime',
    'dischtime',
    'hospital_expire_flag',
    'intime',
    'outtime',
    'ICU_length',
    'Hospital_length']

vitals = {"heart_rate_max":{'itemid':220045, 'agg':'max'}, "blood_pressure_min":{'itemid':220181,"agg":'min'}}

labevents = {"sodium_max":{'itemid':[50983,52623],'agg':'max'}, "sodium_min":{'itemid':[50983,52623],'agg':'min'},"potassium_max":{'itemid':[52610,50971],'agg':'max'},"bun_max":{'itemid':[51006,52647], 'agg':'max'},"creatinine_max":{'itemid':[50912,52546],'agg':'max'},"glucose_min":{'itemid':[50931,52569],'agg':'min'},"pH_min":{'itemid':[50820],'agg':'min'},"lactate_max":{'itemid':[50813, 52442, 53154],'agg':'max'}, "platelet_max":{'itemid':[51704,51265],'agg':'max'},"wbc_max":{'itemid':[51301, 51755, 51756],'agg':'max'},"hemoglobin_min":{'itemid':[50811, 51222, 51640],'agg':'min'},"ast_max":{'itemid':[53088,50878],'agg':'max'},"alt_max":{'itemid':[50861],'agg':'max'},"bilirubin_max":{'itemid':[50885,53089],'agg':'max'},"inr_max":{'itemid':[51675,51237],'agg':'max'}}

antibiotics = ['Vancomycin', 'Piperacillin-Tazobactam', 'Ciprofloxacin', 'Ciprofloxacin HCl', 'Meropenem', 'CefePIME', 'CeftriaXONE', 'MetRONIDAZOLE (FLagyl)', 'CefTRIAXone', 'Acyclovir', 'CefazoLIN', 'Sulfameth/Trimethoprim DS', 'Tobramycin', 'Azithromycin', 'Levofloxacin', 'Ampicillin', 'Erythromycin', 'Clindamycin', 'Aztreonam', 'CeFAZolin', 'moxifloxacin', 'Linezolid', 'Micafungin', 'Sulfamethoxazole-Trimethoprim', 'Doxycycline Hyclate', 'CefTAZidime', 'MetroNIDAZOLE', 'Sulfameth/Trimethoprim SS']

vasoactive_agents = ['Norepinephrine', 'NORepinephrine', 'EPINEPHrine', 'Vasopressin', 'DOPamine']

procedure_keywords = ["ventilation", "endotracheal", "intubation", "mechanical ventilation"]

def clean(title, before, after):
    matched_titles = d_diagnoses[d_diagnoses["long_title"].str.contains(title, case=False,na=False)].copy()
    diagnoses_filtered = diagnoses[diagnoses["icd_code"].isin(matched_titles["icd_code"])].copy()
    patients_info = patients[patients["subject_id"].isin(diagnoses_filtered["subject_id"])].copy().reset_index(drop=True)
    admissions_info = admissions[admissions["hadm_id"].isin(diagnoses_filtered["hadm_id"])].copy().reset_index(drop=True)
    df = patients_info.merge(admissions_info,on="subject_id")
    col = "hadm_id"
    df = df[[col] + [c for c in df.columns if c != col]]
    df = df.merge(icustays,on=["hadm_id","subject_id"],how="left")
    df["outtime"]= pd.to_datetime(df["outtime"])
    df["intime"]=pd.to_datetime(df["intime"])
    df["ICU_length"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    df["admittime"] =pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["Hospital_length"] = (df["dischtime"]-df["admittime"]).dt.total_seconds() / 3600
    df = df[columns]
    return get_vitals(df,before,after)
    
def get_vitals(df, before, after):
    df = df.copy()
    df["end_window"] = (df["intime"] + timedelta(hours=after))
    df["start_window"] = (df["intime"] - timedelta(hours=before))
    c = chartevents.copy()
    c["charttime"] =pd.to_datetime(c["charttime"])
    merged  = c.merge(df[["hadm_id","intime","end_window"]],on="hadm_id", how="right")
    mask =  (merged['intime'] <= merged['charttime']) & (merged['charttime']<=merged["end_window"])
    merged = merged[mask]
    for vital, info in vitals.items():
        test = merged[merged["itemid"]==info["itemid"]].groupby("hadm_id")["valuenum"].agg(info["agg"]).reset_index(name=vital)
        df = df.merge(test, on="hadm_id",how="left")
    return get_labs(df)

def get_labs(df):
    df = df.copy()
    l = labs.copy()
    l["charttime"] = pd.to_datetime(l["charttime"])
    merged  = l.merge(df[["hadm_id","intime","end_window"]],on="hadm_id", how="right")
    mask =  (merged['intime'] <= merged['charttime']) & (merged['charttime']<=merged["end_window"])
    merged = merged[mask]
    for event, info in labevents.items():
        test = merged[merged["itemid"].isin(info["itemid"])].groupby("hadm_id")["valuenum"].agg(info["agg"]).reset_index(name=event)
        df = df.merge(test, on="hadm_id",how="left")
    return get_max_creatinine_bun(df)
        
def get_medications(df):
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"])
    merged = p.merge(df[["hadm_id","intime","end_window"]],on="hadm_id", how="right")
    antibiotics_df = merged[merged["medication"].isin(antibiotics)]
    mask = (antibiotics_df['intime'] <= antibiotics_df['starttime']) & (antibiotics_df['starttime']<=antibiotics_df["end_window"])
    antibiotics_df = antibiotics_df[mask]
    antibiotics_df = antibiotics_df.groupby("hadm_id")["medication"].apply(lambda x: list(x.unique())).reset_index(name="antibiotics")
    df = df.merge(antibiotics_df, on ="hadm_id", how = 'left')
    pp = pharmacy.copy()
    merged = pp.merge(df[["hadm_id","intime","end_window"]],on="hadm_id", how="right")
    vaso_df = merged[merged["medication"].isin(vasoactive_agents)]
    mask = (vaso_df['intime'] <= vaso_df['starttime']) & (vaso_df['starttime']<=vaso_df["end_window"])
    vaso_df = vaso_df[mask]
    vaso_df = vaso_df.groupby("hadm_id")["medication"].apply(lambda x:list(x.unique())).reset_index(name="vasoactive_meds")
    df = df.merge(vaso_df,on="hadm_id",how="left")
    return df
    
def get_max_creatinine_bun(df):
    creatinine = labs[ (labs["itemid"].isin([50912,52546])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_cre = creatinine.groupby("hadm_id")["valuenum"].max().reset_index(name="creatinine_admission_max")
    bun = labs[(labs["itemid"].isin([51006,52647])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_bun = bun.groupby("hadm_id")["valuenum"].max().reset_index(name="bun_admission_max")
    df = df.merge(max_cre, on="hadm_id", how="left")
    df = df.merge(max_bun, on="hadm_id", how="left")
    return get_medications(df)

def get_procedures(df):
    procedures_diagnoses = procedures[procedures["hadm_id"].isin(df["hadm_id"])]
    procedures_diagnoses = procedures_diagnoses.merge(d_procedures,on=["icd_code","icd_version"], how="left")
    procedure_mask = procedures_diagnoses['long_title'].str.contains('|'.join(procedure_keywords), case=False, na=False)
    procedure_procs = procedures_diagnoses[procedure_mask]
    procedure_procs_hadm = procedure_procs["hadm_id"]
    df['vent_or_intubation'] = df['hadm_id'].isin(procedure_procs_hadm).astype(int)
    return df
