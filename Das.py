import streamlit as st
import pandas as pd
from pathlib import Path
url = "Challenge dataset traité.xlsx"
data = pd.read_excel(url)
st.dataframe(data)

