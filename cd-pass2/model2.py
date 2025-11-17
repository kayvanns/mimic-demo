import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

df = pd.read_csv("/Users/kayvans/Documents/mimic-demo/cd-pass2/draft.csv")


df = df.drop_duplicates(subset=['hadm_id'])
df = df.drop(columns=["hadm_id","subject_id","stay_id","admittime","dischtime","end_window","start_window","intime","outtime","first_antibiotic_time","uti","temperature_max_C"], axis=1)
categorical_cols= ['gender', 'race', 'admission_type', 'admission_location']
df2 = df.copy()

binary_cols = ['vent_or_intubation', 'sepsis', 'septic_shock', 'hospital_expire_flag','arf','antibiotics_given','vaso_given']
cont_cols = ['creatinine_admission_max', 'bun_admission_max', 'blood_pressure_min','lactate_max',"anchor_age"]

def discretize_creatinine(x):
    if pd.isna(x): 
        return 99
    if x < 1.0: 
        return 0     
    if x < 2.0: 
        return 1      
    return 2                  

def discretize_bun(x):
    if pd.isna(x):
        return 99
    if x < 20: 
        return 0       
    if x < 40: 
        return 1     
    return 2                  

def discretize_blood_pressure(x):
    if pd.isna(x): 
        return 99
    if x < 70: 
        return 0       
    if x < 90: 
        return 1       
    return 2                

def discretize_lactate(x):
    if pd.isna(x): 
        return 99
    if x < 2: 
        return 0        
    if x < 4: 
        return 1        
    return 2                  

def discretize_age(x):
    if pd.isna(x): 
        return 99
    if x < 40: 
        return 0
    if x < 65: 
        return 1
    if x <89:
        return 2
    return 3                 

df2["creatinine_admission_max"] = df2["creatinine_admission_max"].apply(discretize_creatinine)
df2["bun_admission_max"] = df2["bun_admission_max"].apply(discretize_bun)
df2["blood_pressure_min"] = df2["blood_pressure_min"].apply(discretize_blood_pressure)
df2["lactate_max"] = df2["lactate_max"].apply(discretize_lactate)
df2["anchor_age"] = df2["anchor_age"].apply(discretize_age)


for col in categorical_cols:
    df2[col] = df2[col].astype('category').cat.codes
for col in binary_cols:
    df2[col] = df2[col].astype('int')


core_cols = ['antibiotics_given', 'vaso_given','vent_or_intubation', 'creatinine_admission_max', 'bun_admission_max', 'blood_pressure_min','lactate_max','anchor_age','gender',"race",'hospital_expire_flag', 'septic_shock', 'sepsis', 'arf']
df2=df2[core_cols]

data = df2.to_numpy()
cg = pc(data, indep_test='chisq')
nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()
for i in range(len(nodes)):
    bk.add_forbidden_by_node(nodes[9], nodes[i])
bk.add_forbidden_by_node(nodes[10], nodes[11])
cg_with_background_knowledge = pc(data, indep_test='chisq', background_knowledge=bk)
cg_with_background_knowledge.draw_pydot_graph(labels=core_cols)
pyd = GraphUtils.to_pydot(cg_with_background_knowledge.G,labels=core_cols) 
pyd.write_png("model2_chisq_binning.png")
df2.to_csv("data_processed2.csv", index=False)