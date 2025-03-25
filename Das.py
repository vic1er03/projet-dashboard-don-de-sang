import streamlit as st
import pandas as pd
from pathlib import Path
url = "data_2019_preprocessed.csv"
data = pd.read_csv(url,sep=';')
st.dataframe(data)
st.DataFrame(data)
