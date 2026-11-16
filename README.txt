

1. Create a folder named AI_Network_Security on your computer.
2. Copy app.py and requirements.txt into it.
3. Copy these three downloaded model files into the same folder:
   random_forest_model.pkl
   isolation_forest_model.pkl
   preprocessor.pkl
4. Open Windows Terminal or Command Prompt in this folder.
5. Run: py -m pip install -r requirements.txt
6. Run: py -m streamlit run app.py
7. A browser tab opens. Upload a CSV containing the same feature columns used when the models were trained.

If `py` is unavailable, try `python` in its place. Keep the three .pkl files beside app.py.

This is a CSV analysis prototype, not real-time network monitoring. A `label` column is used only for Random Forest evaluation; it is removed before predictions. Without true labels, evaluation metrics are skipped. Isolation Forest -1 means anomaly and 1 means normal.

If you get a feature-column mismatch, the uploaded CSV must match the training columns. The model files alone do not reveal the training notebook's exact column cleanup choices.
