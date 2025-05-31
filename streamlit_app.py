# streamlit_app.py – KI-Demo mit Schritt-für-Schritt Aufbau und Code-Einblick

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
Willkommen zu unserer interaktiven KI-Demonstration! 

Hier erleben Sie, wie ein einfaches maschinelles Lernmodell in vier aufeinander aufbauenden Schritten funktioniert:

1. Wir laden echte Daten
2. Wir trainieren ein KI-Modell
3. Wir visualisieren, wie die KI denkt
4. Wir lassen Sie selbst ausprobieren, wie die KI entscheidet

Am Ende sehen Sie, wie Ihre Eingaben eine KI-Entscheidung auslösen – fast wie Magie, aber vollständig erklärbar.
""")

# Abschnitt 1: Datensatz laden und anzeigen
with st.expander("📦 Schritt 1: Datensatz laden und betrachten"):
    st.markdown("In diesem Schritt laden wir den berühmten Iris-Datensatz. Jede Zeile entspricht einer Blume mit vier gemessenen Merkmalen. Wir konzentrieren uns hier auf zwei davon, damit die Visualisierung einfach bleibt.")
    data = load_iris(as_frame=True)
    df = data.frame
    st.dataframe(df.head())
    X = df[['sepal length (cm)', 'petal length (cm)']]
    y = df['target']
    with st.expander("🔍 Code anzeigen"):
        st.code("""
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame
X = df[['sepal length (cm)', 'petal length (cm)']]
y = df['target']
        """)

# Abschnitt 2: Modell trainieren
with st.expander("🧪 Schritt 2: Modell trainieren"):
    st.markdown("Wir trainieren einen sogenannten Entscheidungsbaum. Dieser Algorithmus versucht, einfache Wenn-Dann-Regeln zu finden, um die Blume einer Gattung zuzuordnen.")
    model = DecisionTreeClassifier()
    model.fit(X, y)
    st.success("Das Modell wurde erfolgreich trainiert.")
    with st.expander("🔍 Code anzeigen"):
        st.code("""
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)
        """)

# Abschnitt 3: Entscheidungsbaum anzeigen
with st.expander("🌳 Schritt 3: So denkt der Entscheidungsbaum"):
    st.markdown("Hier sehen Sie den Entscheidungsbaum, der im letzten Schritt trainiert wurde. Jeder Knoten stellt eine Entscheidung dar – z. B. ob die Blütenblattlänge kleiner als ein bestimmter Wert ist. So hangelt sich der Baum bis zur Entscheidung durch.")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=data.target_names, filled=True, ax=ax)
    st.pyplot(fig)
    with st.expander("🔍 Code anzeigen"):
        st.code("""
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=data.target_names, filled=True, ax=ax)
st.pyplot(fig)
        """)

# Abschnitt 4: Eigene Eingaben testen
with st.expander("🎯 Schritt 4: Probieren Sie es selbst aus"):
    st.markdown("Jetzt sind Sie dran: Geben Sie zwei Werte ein – die Länge des Kelchblatts (Sepal) und des Blütenblatts (Petal) – und lassen Sie die KI raten, welche Gattung es ist.")

    sepal_length = st.slider("👉 Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), 5.0)
    petal_length = st.slider("👉 Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), 1.5)

    sample = pd.DataFrame([[sepal_length, petal_length]], columns=X.columns)
    prediction = model.predict(sample)[0]
    flower = data.target_names[prediction]

    st.markdown("---")
    st.subheader("✨ Das Ergebnis Ihrer Eingabe:")
    st.markdown(f"<div style='font-size:32px; padding:10px; border-radius:8px; background-color:#f0f0f0; text-align:center;'>🌼 Die KI erkennt: <strong>{flower.upper()}</strong></div>", unsafe_allow_html=True)

    with st.expander("🔍 Code anzeigen"):
        st.code("""
import pandas as pd

sample = pd.DataFrame([[sepal_length, petal_length]], columns=X.columns)
prediction = model.predict(sample)[0]
flower = data.target_names[prediction]
        """)

# Fußnote
st.caption("Hinweis: Diese Demo nutzt bewusst nur zwei von vier Merkmalen des Iris-Datensatzes, um die Funktionsweise eines KI-Modells anschaulich zu machen.")