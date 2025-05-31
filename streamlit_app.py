# streamlit_app.py – KI-Demo mit Schritt-für-Schritt Aufbau

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Seitentitel und Intro
st.set_page_config(page_title="Wie KI denkt", layout="centered")
st.title("🧠 KI-Demonstrator: Wie Maschinen lernen")
st.write("""
In dieser interaktiven Demo bauen wir Schritt für Schritt ein einfaches KI-Modell.
Sie können jeden Schritt einzeln aufklappen, nachvollziehen und mit Daten experimentieren.
""")

# Abschnitt 1: Datensatz laden und anzeigen
with st.expander("📦 Schritt 1: Datensatz laden und betrachten"):
    data = load_iris(as_frame=True)
    df = data.frame
    st.write("Der Iris-Datensatz enthält Merkmale von Blumen:")
    st.dataframe(df.head())
    X = df[['sepal length (cm)', 'petal length (cm)']]
    y = df['target']

# Abschnitt 2: Modell trainieren
with st.expander("🧪 Schritt 2: Modell trainieren"):
    st.write("Wir verwenden einen Entscheidungsbaum. Er lernt einfache Regeln aus den Daten.")
    model = DecisionTreeClassifier()
    model.fit(X, y)
    st.success("Modell wurde erfolgreich trainiert.")

# Abschnitt 3: Vorhersage mit Nutzereingabe
with st.expander("🎯 Schritt 3: Vorhersage mit eigenen Werten"):
    st.write("Geben Sie zwei Werte ein, um eine Blume klassifizieren zu lassen:")
    sepal_length = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), 5.0)
    petal_length = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), 1.5)
    sample = pd.DataFrame([[sepal_length, petal_length]], columns=X.columns)
    prediction = model.predict(sample)[0]
    flower = data.target_names[prediction]
    st.markdown(f"**Vorhergesagte Klasse:** `{flower}`")

# Abschnitt 4: Entscheidungsbaum anzeigen
with st.expander("🌳 Schritt 4: Visualisierung des Entscheidungsbaums"):
    st.write("Der Entscheidungsbaum zeigt, wie das Modell entscheidet.")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=data.target_names, filled=True, ax=ax)
    st.pyplot(fig)

# Fußnote
st.caption("Dieses Modell nutzt nur zwei von vier Merkmalen zur besseren Verständlichkeit.")