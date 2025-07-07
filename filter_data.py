import pandas as pd
import numpy as np


def calculate_athlete_level(row):
    """
    Calcule le niveau de l'athlète basé sur plusieurs critères
    Retourne un score de 1 à 3 (1 = Débutant, 2 = Intermédiaire, 3 = Expert)
    """
    score = 0
    
    # Critère 1: Ancienneté (0-5 points)
    if row['annees_experience'] < 1:
        score += 1
    elif row['annees_experience'] < 2:
        score += 2
    elif row['annees_experience'] < 3:
        score += 3
    elif row['annees_experience'] < 5:
        score += 4
    else:
        score += 5
    
    # Critère 2: Nombre de séances par semaine (0-5 points)
    if row['entrainements_par_semaine'] < 1:
        score += 1
    elif row['entrainements_par_semaine'] < 2:
        score += 2
    elif row['entrainements_par_semaine'] < 3:
        score += 3
    elif row['entrainements_par_semaine'] < 4:
        score += 4
    else:
        score += 5
    
    # Critère 3: Kilomètres par semaine (0-5 points)
    if row['km_moyen_par_semaine'] < 10:
        score += 1
    elif row['km_moyen_par_semaine'] < 20:
        score += 2
    elif row['km_moyen_par_semaine'] < 30:
        score += 3
    elif row['km_moyen_par_semaine'] < 50:
        score += 4
    else:
        score += 5
    
    # Critère 4: Vitesse moyenne (0-5 points)
    if row['vitesse_moyenne'] < 8:
        score += 1
    elif row['vitesse_moyenne'] < 10:
        score += 2
    elif row['vitesse_moyenne'] < 12:
        score += 3
    elif row['vitesse_moyenne'] < 14:
        score += 4
    else:
        score += 5
    
    # Normaliser le score sur 20 points vers un niveau de 1 à 3
    if score <= 8:
        niveau = 1  # Débutant
    elif score <= 14:
        niveau = 2  # Intermédiaire
    else:
        niveau = 3  # Expert
    
    return niveau


def calculate_athlete_stats(df):
    # Assurer que timestamp est bien en datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["vitesse_moyenne"] = (df["distance (m)"] / 1000) / (df["elapsed time (s)"] / 3600)

    # Calcul des colonnes dérivées
    df["annee_semaine"] = df["timestamp"].dt.isocalendar().year.astype(str) + "-" + df["timestamp"].dt.isocalendar().week.astype(str).str.zfill(2)

    # Moyenne des km/semaine
    km_par_semaine = (
        df.groupby(["athlete", "annee_semaine"])["distance (m)"]
        .sum()
        .reset_index()
    )
    km_par_semaine["km"] = km_par_semaine["distance (m)"] / 1000
    moyenne_km_semaine = km_par_semaine.groupby("athlete")["km"].mean().reset_index(name="km_moyen_par_semaine")

    # Moyenne des entraînements/semaine
    seances_par_semaine = df.groupby(["athlete", "annee_semaine"]).size().reset_index(name="nb_seances")
    moyenne_seances = seances_par_semaine.groupby("athlete")["nb_seances"].mean().reset_index(name="entrainements_par_semaine")

    # Expérience en années
    experience = df.groupby("athlete")["timestamp"].agg(["min", "max"]).reset_index()
    experience["annees_experience"] = (experience["max"] - experience["min"]).dt.days / 365

    # Vitesse moyenne par athlète
    vitesse_moyenne = df.groupby("athlete")["vitesse_moyenne"].mean().reset_index(name="vitesse_moyenne")

    # Fusion des résultats
    resume = (
        moyenne_km_semaine
        .merge(moyenne_seances, on="athlete")
        .merge(experience[["athlete", "annees_experience"]], on="athlete")
        .merge(vitesse_moyenne, on="athlete")
    )

    return resume


def process_data_with_original_columns(df):
    print(f"Nombre d'entrées initiales : {len(df)}")
    
    # Filtre 1: Supprimer les entrées avec distance < 3km
    df_filtered = df[df['distance (m)'] >= 3000].copy()
    print(f"Après filtrage distance >= 3km : {len(df_filtered)} entrées (supprimé {len(df) - len(df_filtered)} entrées)")
    
    # Filtre 2: Supprimer les entrées avec sexe indéfini (pas M ou F)
    df_filtered = df_filtered[df_filtered['gender'].isin(['M', 'F'])].copy()
    print(f"Après filtrage sexe défini (M/F) : {len(df_filtered)} entrées (supprimé {len(df[df['distance (m)'] >= 3000]) - len(df_filtered)} entrées)")
    
    # Calculer les statistiques par athlète
    stats_athletes = calculate_athlete_stats(df_filtered)
    
    # Calculer la vitesse instantanée pour chaque entrée
    df_filtered["vitesse_instantanee"] = (df_filtered["distance (m)"] / 1000) / (df_filtered["elapsed time (s)"] / 3600)
    
    # Filtre 3: Supprimer les entrées avec vitesse instantanée > 23km/h
    nb_avant_vitesse_inst = len(df_filtered)
    df_filtered = df_filtered[df_filtered['vitesse_instantanee'] <= 23].copy()
    print(f"Après filtrage vitesse instantanée <= 23km/h : {len(df_filtered)} entrées (supprimé {nb_avant_vitesse_inst - len(df_filtered)} entrées)")
    
    # Supprimer la colonne vitesse_moyenne du DataFrame principal pour éviter la duplication
    df_clean = df_filtered.drop(columns=['vitesse_moyenne'], errors='ignore')
    
    # Fusionner avec les données originales pour avoir toutes les colonnes de base
    # On garde TOUTES les entrées originales avec les statistiques calculées par athlète
    result = df_clean.merge(stats_athletes, on="athlete", how="left")
    
    # Filtre 4: Supprimer les entrées avec vitesse moyenne > 23km/h
    result = result[result['vitesse_moyenne'] <= 23].copy()
    print(f"Après filtrage vitesse moyenne <= 23km/h : {len(result)} entrées (supprimé {len(df_clean.merge(stats_athletes, on='athlete', how='left')) - len(result)} entrées)")
    
    # Convertir les genres en binaire (H = 0, F = 1)
    result['gender_binary'] = result['gender'].map({'M': 0, 'F': 1})
    
    # Calculer le niveau de chaque athlète
    result['niveau'] = result.apply(calculate_athlete_level, axis=1)
    
    return result


if __name__ == "__main__":
    # Charger le fichier CSV
    df = pd.read_csv("./raw_data_kaggle.csv", sep=";", parse_dates=["timestamp"], dayfirst=True)

    # Appeler la fonction d'analyse avec toutes les colonnes originales
    result = process_data_with_original_columns(df)

    # Afficher un aperçu
    print(f"Nombre total d'entrées : {len(result)}")
    print(f"Nombre d'athlètes uniques : {result['athlete'].nunique()}")
    print(f"\nRépartition des niveaux :")
    print(result['niveau'].value_counts().sort_index())
    print(f"\nRépartition des genres (binaire) :")
    print(result['gender_binary'].value_counts().sort_index())
    print("\nAperçu des données avec toutes les colonnes :")
    print(result.head())
    print("\nColonnes disponibles :")
    print(result.columns.tolist())

    # Exporter le résultat dans un fichier
    result.to_csv("./donnees_completes.csv", index=False)
    
    # Exporter aussi le résumé par athlète
    resume = calculate_athlete_stats(df)
    resume.to_csv("./resume_athletes.csv", index=False)
    
    print(f"\nFichiers exportés :")
    print("- donnees_completes.csv : données originales + nouvelles colonnes calculées")
    print("- resume_athletes.csv : résumé par athlète uniquement")
