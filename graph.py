import pandas as pd
import matplotlib.pyplot as plt

# Charger les données avec le bon séparateur
df = pd.read_csv("ultimate_data.csv", sep=';')

# Nettoyer les données : supprimer les lignes avec des valeurs manquantes importantes
print(f"Nombre de lignes avant nettoyage : {len(df)}")
df = df.dropna(subset=['athlete', 'gender_binary', 'niveau', 'annees_experience'])
print(f"Nombre de lignes après nettoyage : {len(df)}")

# Convertir les colonnes numériques (remplacer les virgules par des points)
df['elevation gain (m)'] = pd.to_numeric(df['elevation gain (m)'].str.replace(',', '.'), errors='coerce')
df['vitesse_moyenne'] = pd.to_numeric(df['vitesse_moyenne'].str.replace(',', '.'), errors='coerce')
df['distance (m)'] = pd.to_numeric(df['distance (m)'].str.replace(',', '.'), errors='coerce')

# 📈 Histogramme du dénivelé
plt.figure(figsize=(10, 6))
plt.hist(df['elevation gain (m)'].dropna(), bins=60, color='purple', edgecolor='black')
plt.xlim(0, 2000)
plt.title("Histogramme du Dénivelé")
plt.xlabel("Dénivelé (m)")
plt.ylabel("Nombre d'efforts")
plt.show()

# 📈 Histogramme de la vitesse moyenne
plt.figure(figsize=(10, 6))
plt.hist(df['vitesse_moyenne'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title("Histogramme de la Vitesse Moyenne")
plt.xlabel("Vitesse moyenne (km/h)")
plt.ylabel("Nombre d'efforts")
plt.show()

# (Optionnel) Supprimer les doublons d'athlètes si on veut 1 point par personne
df_unique = df.drop_duplicates(subset='athlete')

print("Valeurs manquantes après nettoyage :")
print(df_unique[['gender_binary', 'niveau', 'annees_experience']].isnull().sum())

# 📊 Répartition Homme / Femme (avec gender_binary)
plt.figure(figsize=(8, 6))
gender_binary_labels = {0: 'Homme', 1: 'Femme'}
df_unique['gender_binary_label'] = df_unique['gender_binary'].replace(gender_binary_labels)
gender_binary_counts = df_unique['gender_binary_label'].value_counts()
gender_binary_counts.plot(kind='bar', color='pink')
for i, v in enumerate(gender_binary_counts):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.title("Répartition Homme / Femme")
plt.xlabel("Sexe")
plt.ylabel("Nombre d'athlètes")
plt.show()

# 📊 Répartition par Niveau
plt.figure(figsize=(10, 6))
# Convertir les niveaux numériques en labels
niveau_labels = {1: 'Débutant', 2: 'Intermédiaire', 3: 'Expert'}
df_unique['niveau_label'] = df_unique['niveau'].replace(niveau_labels)
niveau_counts = df_unique['niveau_label'].value_counts()
niveau_counts.plot(kind='bar', color='orange')
for i, v in enumerate(niveau_counts):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.title("Répartition par Niveau")
plt.xlabel("Niveau")
plt.ylabel("Nombre d'athlètes")
plt.show()

# 📊 Répartition par Ancienneté
plt.figure(figsize=(10, 6))
# Créer une fonction pour catégoriser l'expérience
def categorize_experience(exp_str):
    try:
        # Remplacer la virgule par un point si nécessaire
        if isinstance(exp_str, str):
            exp_str = exp_str.replace(',', '.')
        exp = float(exp_str)
        if exp < 2:
            return "<2 ans"
        elif exp < 5:
            return "2-5 ans"
        elif exp < 10:
            return "5-10 ans"
        elif exp < 20:
            return "10-20 ans"
        else:
            return "20+ ans"
    except (ValueError, TypeError):
        return "Non renseigné"

# Appliquer la catégorisation
experience_cat = df_unique['annees_experience'].apply(categorize_experience)
experience_counts = experience_cat.value_counts(sort=False)
experience_counts.plot(kind='bar', color='green')
for i, v in enumerate(experience_counts):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.title("Répartition par Ancienneté")
plt.xlabel("Années d'expérience")
plt.ylabel("Nombre d'athlètes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 📊 Statistiques supplémentaires
print("\n=== Statistiques générales ===")
print(f"Nombre total d'efforts : {len(df)}")
print(f"Nombre d'athlètes uniques : {len(df_unique)}")
print(f"Vitesse moyenne globale : {df['vitesse_moyenne'].mean():.2f} km/h")
print(f"Dénivelé moyen : {df['elevation gain (m)'].mean():.2f} m")
print(f"Distance moyenne : {df['distance (m)'].mean()/1000:.2f} km")
