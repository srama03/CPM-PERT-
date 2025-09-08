def forward_pass(G, topo_sorted):
    """
    input: the original graph, the sorted graph
    output: 
        dict that maps each task with its earliest start time
        dict that maps each task with its earliest finish time
        the total duration to complete the project
    """
    #storing the es and ef values
    es_dict = {}
    ef_dict = {}
    #for each task calculate es (maximum of the prev efs) and ef (es + duration)
    for task in topo_sorted:
        preds = G.predecessors(task)
        es = max([ef_dict[p] for p in preds], default = 0) #get the largest ef of predecessors
        ef = es+ G.nodes[task]["duration"]
        es_dict[task]= es
        ef_dict[task]= ef
    #calculate total duration
    total_dur = max(ef_dict.values()) 
    return es_dict, ef_dict, total_dur

def backward_pass(G, topo_sorted, total_dur):
    """
    input: the original graph, the sorted graph, total duration to finish project
    output: 
        dict that maps each task with its latest start time
        dict that maps each task with its latest finish time
    """
    #reverse the sorted list
    rev = list(reversed(topo_sorted))
    #for storing the vals
    ls_dict = {}
    lf_dict = {}
    #for each task calculate ls and lf
    for task in rev:
        succs = list(G.successors(task)) #get successors
        valid_ls= [ls_dict[s] for s in succs if s in ls_dict] #get the ls values of only those nodes that have already been processed- guards against ValueError
        lf = min(valid_ls) if valid_ls else total_dur
        lf_dict[task] = lf
        ls= lf - G.nodes[task]["duration"] 
        ls_dict[task] = ls
    return ls_dict, lf_dict

def calculate_slack(G, topo_sorted, es_dict, ls_dict):
    """
    input: 
        the graph 
        sorted graph
        dict mapping task to earliest start time
        dict mapping task to latest start time
    output:
        dict with slack per task
        list outlining critical path
    """
    slack_dict = {}
    cp = []
    for task in topo_sorted:
        ls = ls_dict[task]
        es = es_dict[task]
        slack = ls - es
        slack_dict[task] = slack
        cp.append(task) if slack == 0 else None #add nodes that fall on the critical path to the list
    return slack_dict, cp
    