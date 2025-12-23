# CPM / PERT Streamlit App

An interactive Streamlit application for analyzing project schedules using **Critical Path Method (CPM)** and **Program Evaluation and Review Technique (PERT)**.  
The app allows users to upload task data, compute critical paths, slack times, and visualize scheduling dependencies.

---

## Overview

Project scheduling techniques like CPM and PERT are widely used in operations research and project management to identify bottlenecks, estimate completion times, and assess risk.  
This project implements these techniques in Python and exposes them through a simple **web-based Streamlit interface**.

The goal of this app is to make CPM/PERT analysis more accessible and interactive, while keeping the underlying logic explicit and modular.

---

## Features

- Parse task dependency data from CSV, JSON, or Excel formats  
- Compute:
  - Earliest start / finish times  
  - Latest start / finish times  
  - Slack for each activity  
  - Critical path(s)
- Support both deterministic (CPM) and probabilistic (PERT) duration inputs
- Interactive Streamlit UI for:
  - Uploading input files  
  - Viewing computed schedules  
  - Exploring critical paths and task slack
- Modular Python implementation separating:
  - Graph construction  
  - Scheduling logic  
  - Input parsing  
  - UI layer

---

## Tech Stack

- **Python**
- **Streamlit** (web app framework)
- **Pandas / NumPy** (data handling)
- **Graph-based scheduling logic** (custom implementation)

---

## Project Structure

CPM-PERT/
│
├── app.py # Streamlit application entry point
├── cpm.py # CPM / PERT computation logic
├── graph_helpers.py # Graph utilities for dependency handling
├── input_parser.py # Input file parsing and validation
├── requirements.txt
├── README.md
└── sample_data/ # Example input files


---

## How to Run Locally

1. Clone the repository:
   ```
   git clone https://github.com/srama03/CPM-PERT.git
   cd CPM-PERT
   ```
Create and activate a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```
Run the Streamlit app:
```
streamlit run app.py
```
Open the local URL shown in the terminal to interact with the app.

### Input Format

The app expects task-level project data including:

- Task ID / name

- Predecessor relationships

- Duration (single value for CPM, or optimistic / most likely / pessimistic for PERT)

Sample input files are provided in the repository. Please Note: the visualisations (esp. the network graph) work best for projects with <= 10 tasks.


## Current Status

This project is actively being cleaned and extended.
Upcoming improvements may include:

Improved visualizations for dependency graphs

More robust input validation

Separation of analysis logic from UI for easier testing

Performance optimizations for larger task graphs

## Notes

This project was built as part of an applied exploration of project scheduling and operations research concepts, with an emphasis on clarity, correctness, and usability rather than production-scale deployment.
