import pandas as pd
import re # for parsing ; and , in csvs
def parse_df(df):
    """
    input: dataframe containing task (string), duration (int), and dependencies (list of strings)
    output:
        - duration_dict: {label: duration}
        - edge_list: [(dep_label, task_label), ...]
        - label_to_task: {label: task_name}
    """
    df= df.copy()
    # normalize task names to prevent case-related errors
    df['Task'] = df["Task"].astype(str).str.strip().str.lower()
    #assign labels
    df['Label'] = [chr(65+i) for i in range(len(df))]
    #dict that maps label to task-> for output and visualization
    label_to_task = {k : v for (k,v) in zip(df['Label'], df['Task']) }
    #dict that maps task to label-> for input processsing
    task_to_label = {v : k for (k,v) in label_to_task.items()}
    #dict that maps label to duration
    duration_dict = {k : v for (k,v) in zip(df['Label'], df['Duration'])}
    #list containing the edge pairs (dependencies -> task)
    edge_list = []
    for index, row in df.iterrows():
        task_label = row['Label']
        dep_str = row['Dependencies']
        #check for no values in dep
        if pd.isna(dep_str) or dep_str.strip() == "":
            continue
        try:
            deps = [i.strip().lower() for i in re.split(r'[;,]', dep_str)]
        except e as KeyError:
            st.error(f"Unknown dependency: '{e.args[0]}'. Please check for typos or case mismatches.")
            st.stop()
        dep_labels = [task_to_label[i.strip().lower()] for i in deps]
        edge_list.extend([(dep_label, task_label) for dep_label in dep_labels])
    return duration_dict, edge_list, label_to_task
    

def parse_pert_df(df):
    """
    input: dataframe containing:
        - task (string)
        - optimistic (int)
        - most likely (int)
        - pessimistic (int)
        - dependencies (list of strings)
    output:
        - te_dict: {label: expected time}
        - edge_list: [(dep_label, task_label), ...]
        - label_to_task: {label: task_name}
        - variance_dict: {label: variance}
    """
    df= df.copy()
    #assign labels
    df['Label'] = [chr(65+i) for i in range(len(df))]
    #dict that maps label to task-> for output and visualization
    df['Task'] = df['Task'].astype(str).str.strip().str.lower()
    label_to_task = {label: task for label, task in zip(df['Label'], df['Task'])}
    #dict that maps task to label-> for input processsing
    task_to_label = {task: label for label, task in label_to_task.items()}
    # dict that maps label to expected time, where TE = (O+4M+P)/6
    te_dict ={}
    for index, row in df.iterrows():
        O = row['Optimistic']
        M = row['Most Likely']
        P = row['Pessimistic']
        lab = row['Label']
        te = (O + 4*M+ P)/6
        te_dict[lab] = te
    #list containing the edge pairs (dependencies -> task)
    edge_list = []
    for index, row in df.iterrows():
        task_label = row['Label']
        dep_str = row['Dependencies']
        #check for no values in dep
        if pd.isna(dep_str) or dep_str.strip() == "":
            continue
        deps = [i.strip().lower() for i in re.split(r'[;,]', dep_str)]
        dep_labels = [task_to_label[i.strip().lower()] for i in deps]
        edge_list.extend([(dep_label, task_label) for dep_label in dep_labels])
    # dict that maps labels to variance
    var_dict = {}
    for index, row in df.iterrows():
        O = row['Optimistic']
        M = row['Most Likely']
        P = row['Pessimistic']
        lab = row['Label']
        var = ((P - O)/6)**2
        var_dict[lab] = var
    return te_dict, edge_list, label_to_task, var_dict
    

# == ALIAS MAP FOR SEMANTIC CHECKING ==
alias_map = {
    # task synonyms
    "activity": "task",
    "activities": "task",
    "work": "task",
    "job": "task",
    "item": "task",
    "task name": "task",
    

    # duration synonyms
    "time": "duration",
    "length": "duration",
    "days": "duration",
    "weeks": "duration",
    "period": "duration",
    "estimate": "duration",
    "time req": "task",
    "time required": "task",

    # dependencies synonyms
    "dependency": "dependencies",
    "predecessor": "dependencies",
    "predecessors": "dependencies",
    "depends on": "dependencies",
    "required before": "dependencies",

    # pert synonyms
    "optimistic time": "optimistic",
    "most likely time": "most likely",
    "pessimistic time": "pessimistic",
    "o": "optimistic",
    "m": "most likely",
    "p": "pessimistic",
    "best case": "optimistic",
    "worst case": "pessimistic"
}