# === FRONT MATTER === 
# libraries
import pandas as pd 
import streamlit as st
import numpy as np 
from scipy.stats import norm
import os
import networkx as nx
import matplotlib.pyplot as plt
import difflib      # for fuzzy matching
from io import BytesIO  # to use aas buffer for export options
# functions
from utils.input_parser import parse_df, parse_pert_df, alias_map
from utils.graph_helpers import build_graph
from utils.cpm import forward_pass, backward_pass, calculate_slack

def load_file(uploaded_file):
    """
    input: csv, json, or excel file
    reads file accordingly; generates error in case of an unsupported file format
    output: pandas dataframe
    """
    file_type = uploaded_file.name.split('.')[-1] #retrieves extension to get the file format
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
        elif file_type in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Upload only excel, csv or json files")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# streamlit preview- ui
st.title("Project Planner: CPM & PERT")

# about app sidebar
with st.sidebar.expander("**About This App**", expanded=True):
    st.markdown("""
    **Planning Made Easier.**

    Whether you're managing a class project, organizing an event, or leading a full-scale operation — this app helps you plan with clarity.
    Just upload a CSV, JSON, or Excel file with columns for task, duration (optimistic, realistic, and pessimistic estimates for PERT), and predecessors, and:
    - Map task dependencies
    - Spot what's critical and what has slack
    - Visualize your timeline with Gantt charts, network diagrams, and PERT probability curves
    - Estimate how likely you are to finish on time (PERT)

    **Ideal For:**
    - Project managers, students, and planners managing complex workflows.

    Simple. Visual. Insightful.
    """)


#create tabs for convenience
upload, result, viz = st.tabs(["Upload", "Results", "Visualizations"])
# === File Upload & Configuration ===
with upload:
    # add toggle for using pert instead of cpm
    use_pert = st.toggle("Use PERT (3-point estimates)")
    uploaded_file = st.file_uploader("Upload task file (.csv, .json, or .xlsx)", type= ["csv", "json", "xlsx"])
    unit = st.selectbox("Select time unit for duration: ", [" ", r"hours", "days", "weeks", "months"])
    if uploaded_file:
        df =  load_file(uploaded_file)
        if df is not None:
            # normalize for mapping
            df.columnrs = df.columns.str.strip()
            df.rename(columns=lambda col: alias_map.get(col.lower(), col), inplace=True)

            # === Data Validation ===
            # check for required cols
            # normalize for validation
            df.columns = df.columns.str.title()
            
            req_cpm = {"Task", "Duration", "Dependencies"}
            req_pert = {"Task", "Optimistic", "Most Likely", "Pessimistic", "Dependencies"}
            req = req_pert if use_pert else req_cpm
            if not req.issubset(df.columns):
                missing = req - set(df.columns)
                suggestions = {}
                for col in missing:
                    close = difflib.get_close_matches(col,df.columns, n=1, cutoff=0.6)
                    if close:
                        suggestions[col] = close[0]
                msg = f"Missing required column(s): {', '.join(sorted(missing))}" # msg if no close matches
                # alias mapping
                if suggestions:
                    msg+="\n\nDid you mean:\n"
                    for req, sug in suggestions.items():
                        msg+=f"- '{req}' instead of '{sug}'?\n"
                st.error(msg)
                st.stop()
            # check if OPM/duration is numeric and non-negative for both cases (cpm, pert)
            if use_pert:
                for col in ["Optimistic", "Pessimistic", "Most Likely"]:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        st.error(f"The '{col}' column should contain only numbers.")
                        st.stop()
                    if (df[col]<0).any():
                        st.error(f"{col} must be non-negative")
                        st.stop()
            else:
                if not pd.api.types.is_numeric_dtype(df["Duration"]):
                    st.error("The 'Duration' column should contain only numbers.")
                    st.stop()
                if (df["Duration"]<0).any():
                    st.error("Duration must be non-negative")
                    st.stop()
            # check for duplicate tasks
            if df["Task"].duplicated().any():
                dupes = df[df["Task"].duplicated(keep=False)]["Task"].unique() # use a 'mask' to reduce scope of df
                st.warning(f"Duplicate tasks found: {','.join(dupes)}")
            # validate dependency names- check for typos/mismatch of task names
            tasks = set(df["Task"])
            invalid_deps = []
            for i, deps in enumerate(df["Dependencies"].fillna("")):
                for dep in map(str.strip, deps.rsplit(',')):
                    if dep and dep not in tasks: # exclude the empty strings
                        invalid_deps.append((i+1, dep)) # i+1 = row index
            if invalid_deps:
                unique_deps = sorted(set(d for _, d in invalid_deps))
                st.warning(f"Unrecognized dependencies found: {', '.join(set(unique_deps))}")

            # === Preview Dataset Info ===
            st.info(f"**Preview:** ({df.shape[0]} rows x {df.shape[1]} columns)") 
            with st.expander("Show full dataset"):
                st.dataframe(df) # gives preview of uploaded data as a table
            # warning regarding performance for larger files
            if df.shape[0]>500:
                st.info("Large project detected- loading may take some time!")
            if df.shape[1]>1000:
                st.warning("Dataset has more than 1000 tasks. This may affect performance.")
            st.markdown("---")

            
            # get relevant dicts using parse_df
            st.markdown("### Parsing input...")
            # === Parse Input Based on Selected Mode ===
            # parse accordingly- cpm, pert
            
            if use_pert:
                duration_dict, edge_list, label_to_task, var_dict = parse_pert_df(df)
                st.session_state["var_dict"] = var_dict
            else:
                duration_dict, edge_list, label_to_task = parse_df(df)
                st.session_state['var_dict'] = None
            
            st.success("Successfully parsed!")

    # === Debugging: Show Internal Mappings ===
    with st.expander("See internal mappings (for debugging)"):
        if uploaded_file and df is not None:
            st.markdown("**Task-Duration mapping:**")
            st.table(pd.DataFrame(list(duration_dict.items()),columns = ["Task", "Duration"]))

            st.markdown("**Dependency order:** ")
            st.json(edge_list)

            st.markdown("**Task-Label mapping:** ")
            st.json(label_to_task)
    
    # === CPM/PERT Result Calculations ===
    with result:
        if uploaded_file and df is not None:
            # build the graph 
            G = build_graph(duration_dict, edge_list)
            # topological sort
            topo_sorted = list(nx.topological_sort(G))
            # call core logic
            # forward pass
            es_dict, ef_dict, total_dur = forward_pass(G, topo_sorted)
            # backward pass
            ls_dict, lf_dict = backward_pass(G,topo_sorted, total_dur)
            # calculate slack and critical path
            slack_dict, cp = calculate_slack(G, topo_sorted, es_dict, ls_dict)
            
            # === Summary Table (ES/EF/LS/LF/Slack) ===
            rows = []
            for label in topo_sorted:
                task_name = label_to_task[label] #get actual task 
                rows.append({
                    "Task" : task_name,
                    "ES" : round(es_dict[label],2),
                    "EF" : round(ef_dict[label], 2),
                    "LS" : round(ls_dict[label], 2),
                    "LF" : round(lf_dict[label], 2),
                    "Slack" : round(slack_dict[label], 2) if slack_dict[label] > 1e-2 else 0,
                    "Critical?" : "Yes" if label in cp else "No"
                })

            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df)

            st.markdown(f"Total Duration for Project: {total_dur:.2f} {unit} ")

            if use_pert and "var_dict" in st.session_state:
                var_dict = st.session_state["var_dict"] # get var_dict
                # find project var and sd using only cp nodes
                cp_var = sum(var_dict[n] for n in cp)
                cp_sd = cp_var ** 0.5
                # contextual interpretation
                st.markdown("### PERT Analysis- Uncertainty Estimation")
                st.write(
                    f"Based on variance in time estimates for tasks on the critical path, "
                    f"the standard deviation of project duration is ±{cp_sd:.2f} {unit}."
                )
                st.success(
                    f"You can be ~68% confident that the project will complete within "
                    f"{total_dur:.2f} ± {cp_sd:.2f} {unit}, "
                    f"and ~95% confident within {total_dur:.2f} ± r{2*cp_sd:.2f} {unit}."
                )
                st.markdown(
                    f"_This means there is some uncertainty in project timing. The more uncertain the task durations, the more variation in your final delivery date._ "
                )
                # prob of finishing project within custom deadline
                with st.expander("Calculate probability for a custom deadline: "):
                    mu = total_dur
                    sigma = cp_sd
                    st.markdown(
                        "Enter your target completion duration and see how likely it"
                        "is to finish within that timeframe based on the PERT analysis."
                    )
                    deadline = st.number_input(
                        f"Target completion time ({unit}): ",
                        min_value=0.0,
                        value=round(mu+sigma, 2),
                        step=0.1
                        )
                    if deadline:
                        prob = norm.cdf(deadline, loc=mu, scale=sigma)
                        st.success(
                            f"Based on your PERT estimates, there is a **{prob*100:.1f}%** chance "
                            f"that the project will complete within **{round(deadline,2)} {unit}**."
                        )
            else:
                st.info(
                    "The **total project duration** is based on the longest sequence of dependent tasks "
                    "(the critical path). Any delay in these tasks will directly affect the project completion time."
                )
                st.success(
                    "Tasks with **0 slack** are critical."
                )
            
    # === Visualizations ===
    with viz:
        if uploaded_file and df is not None:
            if use_pert:
                # --- PDF ---
                mu = total_dur
                sigma = cp_sd
                # x range: mean +- 4sig
                x_range = np.linspace(mu-4*sigma, mu+4*sigma, 500)
                # pdf vals
                pdf_vals = norm.pdf(x_range, loc=mu, scale=sigma)
                # plot
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(x_range, pdf_vals, color='#4A90E2', lw=2)
                # +- sigma: 68% range
                ax.axvline(mu - sigma, color='#1a1a1a', linestyle='--', lw=1)
                ax.axvline(mu + sigma, color='#1a1a1a', linestyle='--', lw=1)
                ax.fill_between(
                    x_range, 
                    pdf_vals, 
                    where=((x_range>=mu-sigma) & (x_range<=mu+sigma)),
                    color="#BBDEFB",
                    alpha=0.7,
                    label="~68% range"
                    )
                # +- 2sigma: 95% range
                ax.axvline(mu - 2*sigma, color='#1a1a1a', linestyle=':', lw=1)
                ax.axvline(mu + 2*sigma, color='#1a1a1a', linestyle=':', lw=1)
                ax.fill_between(
                    x_range, 
                    pdf_vals, 
                    where=((x_range>=mu-2*sigma) & (x_range<=mu+2*sigma)),
                    color="#E3F2FD",
                    alpha=0.5,
                    label="~95% range"
                    )
                # mean
                ax.axvline(mu, color='tomato', linestyle='-', lw=2, label='Expected Duration')
                # labels
                ax.set_title("Project Completion Probability Distribution")
                ax.set_xlabel(f"Project Duration ({unit})")
                ax.set_ylabel("Probability Density")
                ax.legend()
                # display
                st.pyplot(fig)
                # option to export
                buffer = BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                st.download_button(
                    label="Download Chart as PNG",
                    data=buffer.getvalue(),
                    file_name="prob_dist.png", 
                    mime="image/png"
                )
            else: # for cpm
                c1, c2 = st.columns(2)
                with c1:
                    gantt_bool = st.toggle("Gantt Chart")
                with c2:
                    only_cp = st.toggle("Show only critical path")
                # task separation based on toggle choice
                tasks_to_plot = cp if only_cp else topo_sorted
                # create dataframe for legend
                legend_df = pd.DataFrame({
                    "Label" : list(label_to_task.keys()),
                    "Task" : [label_to_task[k] for k in label_to_task]
                })
                # --- Gantt Chart ---
                if gantt_bool: 
                    fig, ax = plt.subplots(figsize=(8, 4), facecolor = 'whitesmoke')
                    ax.set_facecolor("whitesmoke")
                    for i,task in enumerate(tasks_to_plot):
                        start = es_dict[task]
                        duration= ef_dict[task] - start
                        color = "#EF9A9A" if slack_dict[task] == 0 else "#BBDEFB"
                        ax.barh(
                            y=i,                #row number
                            width= duration,    #how long the task takes (the width of each bar)
                            left= start,        #where the bar starts on the x-axis
                            height= 0.4,        #thickness
                            color= color,       #red or blue
                            edgecolor= "#1a1a1a"  #outline of the bar
                        )
                        task_name = label_to_task[task]
                        if duration < 1.2:
                            # outside the bar
                            text_x = start + duration + 0.1
                            # get axis max so label doesn't fly off
                            x_max = ax.get_xlim()[1]
                            if text_x> x_max:
                                text_x = start + duration - 0.3  # pull label back inside
                            ax.text(
                                text_x, 
                                i,                                          # same row
                                f"s ={slack_dict[task]}",                   # slack
                                ha="left", va="center",                     # alignment
                                color="#1a1a1a",                   
                                fontsize=6, fontweight="medium"
                                )
                        else:
                            # inside the bar
                            ax.text(
                                start + 0.1, 
                                i,                                          # same row
                                f"s ={slack_dict[task]}",                   # slack
                                ha="left", va="center",                     # alignment
                                color="#1a1a1a", 
                                fontsize=6, fontweight="medium"
                                )
                    ax.set_xlim(0, total_dur + 1)    # expand x-axis for some breathing space
                    ax.set_yticks(range(len(topo_sorted)))
                    ax.set_yticklabels(topo_sorted)
                    ax.set_xlabel("Time")
                    ax.set_title("Gantt Chart with Critical Tasks")
                    plt.grid(axis="x", linestyle=":", color= "gray", alpha=0.5)
                    st.pyplot(fig)
                    # export
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    st.download_button(
                        label="Download Chart as PNG",
                        data=buffer.getvalue(),
                        file_name="gantt_chart.png",  
                        mime="image/png"
                    )
                    # --- Label <-> Task Legend ---
                    with st.expander("Legend"):
                        st.dataframe(legend_df)

                # --- Network Diagram ---  
                else: 
                    if only_cp:
                        # Filter to only CP nodes and consecutive edges
                        sub_nodes = cp
                        sub_edges = [(u, v) for u, v in G.edges() if u in cp and v in cp and cp.index(v) == cp.index(u) + 1]
                        G_sub = G.edge_subgraph(sub_edges).copy()
                    else:
                        G_sub = G.copy()
                    pos = nx.spring_layout(G_sub, seed=42)
                    node_colors = ["#EF9A9A" if node in cp else "#BBDEFB" for node in G_sub.nodes()] #define colors for nodes
                    edge_colors = ["#E57373" if u in cp and v in cp and cp.index(v)== cp.index(u) + 1 else "#64B5F6" for u,v in G_sub.edges()] #edges: both nodes should be in cp; maintain order
                    fig, ax = plt.subplots(figsize = (8, 6), facecolor = "whitesmoke")
                    ax.set_facecolor("whitesmoke")
                    
                    nx.draw(
                            G_sub,
                            pos,
                            with_labels=True,
                            node_color= node_colors,
                            edge_color= edge_colors,
                            node_size= 1500,
                            font_weight= "bold",
                            font_color= "#1a1a1a",
                            ax= ax
                            )
                    # --- Label <-> Task Legend ---
                    labels = {node: f"\n\ns={slack_dict[node]}" for node in G_sub.nodes}
                    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=9, horizontalalignment="center",  ax=ax)
                    ax.set_title("Network Diagram with Critical Path")
                    st.pyplot(fig)
                    # export option
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    st.download_button(
                        label="Download Chart as PNG",
                        data=buffer.getvalue(),
                        file_name="dag_network.png",  
                        mime="image/png"
                    )
                    
                    # add expander to show legend
                    with st.expander("Legend"):
                        st.dataframe(legend_df)


               





