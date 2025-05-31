# streamlit_app.py – Didaktisch schöne KI-Demo mit Iris-Datensatz

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Seite konfigurieren
st.set_page_config(page_title="Wie KI denkt", layout="centered")
st.title("🧠 KI-Demonstrator: Wie Maschinen lernen")
st.write("""
Willkommen! In dieser Demo sehen Sie, wie ein Entscheidungsbaum kleine Regeln aus Daten lernt.
Sie können unten zwei Werte verändern – und sofort sehen, welche Blume die KI erkennt.
""")

# Datensatz laden und vorbereiten
data = load_iris(as_frame=True)
df = data.frame
X = df[['sepal length (cm)', 'petal length (cm)']]
y = df['target']

# Modell trainieren
model = DecisionTreeClassifier()
model.fit(X, y)

# Benutzer-Eingabe mit Slidern
st.header("🔢 Eingabewerte festlegen")
sepal_length = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), 5.0)
petal_length = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), 1.5)

# Vorhersage treffen
sample = pd.DataFrame([[sepal_length, petal_length]], columns=X.columns)
pred = model.predict(sample)[0]
flower = data.target_names[pred]

# Ergebnis anzeigen
st.subheader("🌼 Ergebnis")
st.markdown(f"**Vorhergesagte Klasse:** `{flower}`")

# Entscheidungsbaum anzeigen
st.subheader("🌳 Wie der Entscheidungsbaum denkt")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=data.target_names, filled=True, ax=ax)
st.pyplot(fig)

# Fußnote
st.caption("Dieses Modell verwendet nur zwei der vier Iris-Merkmale, um es besser verständlich zu machen.")