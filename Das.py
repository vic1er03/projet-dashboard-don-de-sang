import streamlit as st
import pandas as pd
from pathlib import Path
url = "Challenge dataset traité.xlsx"
data = pd.read_csv(url,sep=';')
st.dataframe(data)

