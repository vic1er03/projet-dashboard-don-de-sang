import streamlit as st
import pandas as pd
from pathlib import Path
url = "Challenge_dataset_traité.csv"
data = pd.read_csv(url)
st.dataframe(data)

