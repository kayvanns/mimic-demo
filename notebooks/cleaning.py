import pandas as pd
from datetime import timedelta
import datetime as dt

path = "/Users/kayvans/Downloads/mimic-iv-clinical-database-demo-2.2/hosp/"
path2 = "/Users/kayvans/Downloads/mimic-iv-clinical-database-demo-2.2/icu/"

files = ["admissions", "patients", "labevents", "d_labitems", "prescriptions","pharmacy","transfers", "diagnoses_icd","d_icd_diagnoses","procedures_icd","d_icd_procedures","omr"]
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
omr = dfs["omr"]

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

vitals = {"heart_rate_max":{'itemid':220045, 'agg':'max'}, "blood_pressure_min":{'itemid':220181,"agg":'min'},"spO2_min":{'itemid':220277,'agg':'min'},"FiO2_max":{'itemid':223835, 'agg':'max'},"temperature_max_C":{'itemid':223762, 'agg':'max'},"temperature_max_F":{'itemid':223761,'agg':'max'},"gsc_motor_min":{'itemid':223901,'agg':'min'},"gsc_verbal_min":{'itemid':223900,'agg':'min'},"gsc_eye_min":{'itemid':220739,'agg':'min'}}

labevents = {"sodium_max":{'itemid':[50983,52623],'agg':'max'}, "sodium_min":{'itemid':[50983,52623],'agg':'min'},"potassium_max":{'itemid':[52610,50971],'agg':'max'},"bun_max":{'itemid':[51006,52647], 'agg':'max'},"creatinine_max":{'itemid':[50912,52546],'agg':'max'},"glucose_min":{'itemid':[50931,52569],'agg':'min'},"pH_min":{'itemid':[50820],'agg':'min'},"lactate_max":{'itemid':[50813, 52442, 53154],'agg':'max'}, "platelet_max":{'itemid':[51704,51265],'agg':'max'},"wbc_max":{'itemid':[51301, 51755, 51756],'agg':'max'},"hemoglobin_min":{'itemid':[50811, 51222, 51640],'agg':'min'},"ast_max":{'itemid':[53088,50878],'agg':'max'},"alt_max":{'itemid':[50861],'agg':'max'},"bilirubin_max":{'itemid':[50885,53089],'agg':'max'},"inr_max":{'itemid':[51675,51237],'agg':'max'}}

antibiotics = ['Vancomycin', 'Piperacillin-Tazobactam', 'Ciprofloxacin', 'Ciprofloxacin HCl', 'Meropenem', 'CefePIME', 'CeftriaXONE', 'MetRONIDAZOLE (FLagyl)', 'CefTRIAXone', 'Acyclovir', 'CefazoLIN', 'Sulfameth/Trimethoprim DS', 'Tobramycin', 'Azithromycin', 'Levofloxacin', 'Ampicillin', 'Erythromycin', 'Clindamycin', 'Aztreonam', 'CeFAZolin', 'moxifloxacin', 'Linezolid', 'Micafungin', 'Sulfamethoxazole-Trimethoprim', 'Doxycycline Hyclate', 'CefTAZidime', 'MetroNIDAZOLE', 'Sulfameth/Trimethoprim SS']

vasoactive_agents = ['Norepinephrine', 'NORepinephrine', 'EPINEPHrine', 'Vasopressin', 'DOPamine']

icd_codes_septic_shock = ["R6521","78552"]
icd_codes_sepsis = ["R6520","99592","99591","A41"]
icd_codes_kidney = ["N17","584"]

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
    return get_medications(df)
        
def get_medications(df):
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"])
    merged = p.merge(df[["hadm_id","intime","end_window"]],on="hadm_id", how="right")
    antibiotics_df = merged[merged["medication"].isin(antibiotics)]
    mask = (antibiotics_df['intime'] <= antibiotics_df['starttime']) & (antibiotics_df['starttime']<=antibiotics_df["end_window"])
    antibiotics_df = antibiotics_df[mask]
    ab_flag = antibiotics_df.groupby("hadm_id").size().rename("antibiotics_given") > 0
    vaso_df = merged[merged["medication"].isin(vasoactive_agents)]
    mask = (vaso_df['intime'] <= vaso_df['starttime']) & (vaso_df['starttime']<=vaso_df["end_window"])
    vaso_df = vaso_df[mask]
    vaso_flag = vaso_df.groupby("hadm_id").size().rename("vaso_given") > 0
    df = df.merge(ab_flag, on="hadm_id", how="left")
    df = df.merge(vaso_flag, on="hadm_id", how="left")
    df["antibiotics_given"] = df["antibiotics_given"].fillna(False)
    df["vaso_given"] = df["vaso_given"].fillna(False)
    return df
    
def get_max_creatinine_bun(df):
    creatinine = labs[ (labs["itemid"].isin([50912,52546])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_cre = creatinine.groupby("hadm_id")["valuenum"].max().reset_index(name="creatinine_admission_max")
    bun = labs[(labs["itemid"].isin([51006,52647])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_bun = bun.groupby("hadm_id")["valuenum"].max().reset_index(name="bun_admission_max")
    df = df.merge(max_cre, on="hadm_id", how="left")
    df = df.merge(max_bun, on="hadm_id", how="left")
    return df

def get_time_to_first_antibiotic(df):
    df = df.copy()
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"])
    merged = p.merge(df[["hadm_id","admittime"]],on="hadm_id", how="right")
    antibiotics_df = merged[merged["medication"].isin(antibiotics)]
    mask = antibiotics_df["starttime"] >= antibiotics_df["admittime"]
    antibiotics_df = antibiotics_df[mask]
    first = antibiotics_df.groupby("hadm_id")["starttime"].min().reset_index(name="first_antibiotic_time")
    df = df.merge(first, on="hadm_id", how="left")
    df["time_to_first_antibiotic_hrs"] = (df["first_antibiotic_time"] - df["admittime"]).dt.total_seconds() / 3600
    return df

def get_procedures(df):
    procedures_diagnoses = procedures[procedures["hadm_id"].isin(df["hadm_id"])]
    procedures_diagnoses = procedures_diagnoses.merge(d_procedures,on=["icd_code","icd_version"], how="left")
    procedure_mask = procedures_diagnoses['long_title'].str.contains('|'.join(procedure_keywords), case=False, na=False)
    procedure_procs = procedures_diagnoses[procedure_mask]
    procedure_procs_hadm = procedure_procs["hadm_id"]
    df['vent_or_intubation'] = df['hadm_id'].isin(procedure_procs_hadm).astype(int)
    return df
    
def get_bmi(df):
    o = omr.copy()
    o["chartdate"] = pd.to_datetime(o["chartdate"])
    o = o[o["result_name"].isin(["Height (Inches)", "Weight (Lbs)"])]
    merged = o.merge(df[["subject_id", "hadm_id", "admittime"]],on="subject_id", how="right")
    merged = merged[merged["chartdate"] >= merged["admittime"]]
    pivoted = merged.pivot_table(index=["subject_id", "hadm_id", "chartdate"], columns="result_name",values="result_value", aggfunc="first").reset_index()
    pivoted = pivoted.sort_values(["hadm_id", "chartdate"]).groupby("hadm_id").first().reset_index()
    pivoted["Height (Inches)"] = pd.to_numeric(pivoted["Height (Inches)"], errors="coerce")
    pivoted["Weight (Lbs)"] = pd.to_numeric(pivoted["Weight (Lbs)"], errors="coerce")
    pivoted["BMI"] = (pivoted["Weight (Lbs)"] / (pivoted["Height (Inches)"] ** 2))*703
    pivoted = pivoted[["hadm_id", "BMI"]]
    df = df.merge(pivoted, on="hadm_id", how="left")
    return df

def get_diagnosis_flags(df):
    dx = diagnoses.copy()
    dx["septic_shock"] = dx["icd_code"].str.startswith(tuple(icd_codes_septic_shock)).astype(int)
    dx["sepsis"] = dx["icd_code"].str.startswith(tuple(icd_codes_sepsis)).astype(int)
    dx["arf"] = dx["icd_code"].str.startswith(tuple(icd_codes_kidney)).astype(int)
    dx = dx.groupby("hadm_id")[["septic_shock","sepsis","arf"]].max().reset_index()
    df["uti"] = 1
    return df.merge(dx,on="hadm_id",how="left")
    