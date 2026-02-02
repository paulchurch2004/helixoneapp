# Guide Complet de Modélisation du Risque de Crédit

## Calcul de PD, LGD, EAD et EL avec Machine Learning en Python

---

# Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Glossaire des Acronymes](#2-glossaire-des-acronymes)
3. [Préparation des Données](#3-préparation-des-données)
4. [Modèle PD - Probabilité de Défaut](#4-modèle-pd---probabilité-de-défaut)
5. [Scorecard et Seuils de Décision](#5-scorecard-et-seuils-de-décision)
6. [Modèle LGD - Perte en Cas de Défaut](#6-modèle-lgd---perte-en-cas-de-défaut)
7. [Modèle EAD - Exposition au Défaut](#7-modèle-ead---exposition-au-défaut)
8. [Calcul de la Perte Attendue (EL)](#8-calcul-de-la-perte-attendue-el)
9. [PSI - Indice de Stabilité de Population](#9-psi---indice-de-stabilité-de-population)
10. [Code Complet et Fonctions Utilitaires](#10-code-complet-et-fonctions-utilitaires)

---

# 1. Introduction et Contexte

## 1.1 Qu'est-ce que le Risque de Crédit ?

Le **risque de crédit** représente la probabilité qu'un emprunteur ne soit pas en mesure de rembourser son prêt (crédit immobilier, carte de crédit, prêt personnel, etc.). Dans certains cas, l'emprunteur peut rembourser une partie seulement de la dette, laissant le principal et les intérêts impayés.

**Exemple concret** : Une banque accorde un prêt de 10 000 € à un client. Le risque de crédit évalue la probabilité que ce client ne rembourse pas les 10 000 € plus les intérêts.

## 1.2 Cadre Réglementaire - Accords de Bâle

Les **Accords de Bâle** (Basel Accords en anglais) sont des réglementations bancaires internationales qui définissent comment les banques doivent évaluer et gérer leurs risques.

Il existe deux approches **IRB (Internal Rating Based, approche basée sur les notations internes)** :

| Approche | Description |
|----------|-------------|
| **Foundation IRB** (IRB de base) | La banque estime la PD (Probabilité de Défaut) en interne, mais la LGD (Perte en Cas de Défaut) et l'EAD (Exposition au Défaut) sont prescrites par le régulateur |
| **Advanced IRB** (IRB avancée) | La banque peut estimer en interne la PD, la LGD et l'EAD |

## 1.3 Les Trois Piliers de la Modélisation

### PD (Probability of Default, Probabilité de Défaut)

La **PD** est la probabilité qu'un emprunteur fasse défaut sur sa dette sur une période d'un an.

**Exemple** : Si la PD d'un client est de 5%, cela signifie qu'il y a 5% de chances que ce client ne rembourse pas son prêt dans l'année.

- Exprimée en pourcentage (0% à 100%)
- Plus la probabilité est élevée, plus le risque de défaut est grand
- Modélisée par une **régression logistique**

### LGD (Loss Given Default, Perte en Cas de Défaut)

La **LGD** représente le montant que la banque s'attend à perdre lorsqu'un emprunteur fait défaut.

**Formule** : `LGD = 1 - Taux de Récupération`

**Exemple** : Sur un prêt de 10 000 € en défaut, si la banque récupère 6 000 € (via saisie, vente d'actifs, etc.), alors :
- Taux de récupération = 6 000 / 10 000 = 60%
- LGD = 1 - 0.60 = 40%

### EAD (Exposure at Default, Exposition au Défaut)

L'**EAD** est le montant que l'emprunteur doit à la banque au moment du défaut.

**Exemple** : Un client a un crédit de 10 000 € et a déjà remboursé 3 000 €. S'il fait défaut maintenant, l'EAD = 7 000 €.

### EL (Expected Loss, Perte Attendue)

La **perte attendue** est calculée par la formule :

```
EL = PD × LGD × EAD
```

**Exemple numérique** :
- PD = 5% (0.05)
- LGD = 40% (0.40)
- EAD = 7 000 €

```
EL = 0.05 × 0.40 × 7 000 = 140 €
```

La banque doit provisionner 140 € pour ce prêt.

## 1.4 Types de Modèles de Crédit

| Type | Description | Utilisation |
|------|-------------|-------------|
| **Modèle d'application** | Évalue si un nouveau client doit recevoir un crédit | Décision d'octroi, tarification du risque |
| **Modèle comportemental** | Évalue si un client existant mérite plus de crédit | Gestion des limites, détection précoce |

## 1.5 Dataset Utilisé

Les données proviennent de **Lending Club**, une grande entreprise américaine de prêt entre particuliers (P2P, Peer-to-Peer lending).

- **Volume** : Plus de 800 000 prêts à la consommation
- **Période** : 2007 à 2015
- **Source** : [Kaggle](https://www.kaggle.com/wendykan/lending-club-loan-data/version/1)

**Stratégie de découpage** :
- **Données d'entraînement** : 2007-2014 (construction des modèles)
- **Données de test** : 2015 (validation de la stabilité)

---

# 2. Glossaire des Acronymes

| Acronyme | Signification Anglaise | Signification Française |
|----------|------------------------|-------------------------|
| **PD** | Probability of Default | Probabilité de Défaut |
| **LGD** | Loss Given Default | Perte en Cas de Défaut |
| **EAD** | Exposure at Default | Exposition au Défaut |
| **EL** | Expected Loss | Perte Attendue |
| **IRB** | Internal Rating Based | Approche Basée sur les Notations Internes |
| **WoE** | Weight of Evidence | Poids de l'Évidence |
| **IV** | Information Value | Valeur d'Information |
| **ROC** | Receiver Operating Characteristic | Caractéristique de Fonctionnement du Récepteur |
| **AUC** | Area Under Curve | Aire Sous la Courbe |
| **AUROC** | Area Under ROC | Aire Sous la Courbe ROC |
| **KS** | Kolmogorov-Smirnov | Test de Kolmogorov-Smirnov |
| **PSI** | Population Stability Index | Indice de Stabilité de Population |
| **CCF** | Credit Conversion Factor | Facteur de Conversion du Crédit |
| **DTI** | Debt-to-Income ratio | Ratio Dette sur Revenu |
| **P2P** | Peer-to-Peer | Prêt entre Particuliers |
| **ML** | Machine Learning | Apprentissage Automatique |
| **SSE** | Sum of Squared Errors | Somme des Erreurs au Carré |
| **SE** | Standard Error | Erreur Standard |
| **FIM** | Fisher Information Matrix | Matrice d'Information de Fisher |

---

# 3. Préparation des Données

## 3.1 Importation des Bibliothèques

```python
# ============================================================
# IMPORTS NÉCESSAIRES
# ============================================================

# Manipulation de données
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # Style Seaborn par défaut

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score

# Statistiques
import scipy.stats as stat

# Sauvegarde des modèles
import pickle

# Téléchargement depuis Google Drive
import gdown
```

## 3.2 Chargement des Données

```python
def telecharger_donnees_gdrive(url: str, output: str) -> None:
    """
    Télécharge un fichier depuis Google Drive.
    
    Args:
        url: Lien de partage Google Drive
        output: Nom du fichier de sortie
        
    Exemple:
        >>> telecharger_donnees_gdrive(
        ...     "https://drive.google.com/file/d/ABC123/view",
        ...     "loan_data.csv"
        ... )
    """
    url_clean = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    gdown.download(url_clean, output, quiet=False)

# Chargement du dataset
loan_data = pd.read_csv("loan_data_2007_2014.csv")

print(f"Dimensions du dataset: {loan_data.shape}")
print(f"Nombre de lignes: {loan_data.shape[0]:,}")
print(f"Nombre de colonnes: {loan_data.shape[1]}")
```

## 3.3 Variables du Modèle

### Variables Discrètes (Catégorielles)

| Variable | Description | Exemple |
|----------|-------------|---------|
| `grade` | Note du prêt attribuée | A, B, C, D, E, F, G |
| `sub_grade` | Sous-note du prêt | A1, A2, B3, etc. |
| `home_ownership` | Statut de propriété | RENT, OWN, MORTGAGE, OTHER |
| `addr_state` | État de résidence | CA, NY, TX, etc. |
| `verification_status` | Statut de vérification | Verified, Not Verified |
| `purpose` | Objet du prêt | debt_consolidation, credit_card, etc. |
| `initial_list_status` | Statut initial de listing | W (Whole), F (Fractional) |

### Variables Continues (Numériques)

| Variable | Description | Exemple |
|----------|-------------|---------|
| `term` | Durée du prêt en mois | 36, 60 |
| `emp_length` | Années d'emploi | 0 à 10+ |
| `int_rate` | Taux d'intérêt | 5.42%, 12.5% |
| `annual_inc` | Revenu annuel déclaré | $50,000, $120,000 |
| `dti` | DTI (Debt-to-Income, ratio dette/revenu) | 15.5, 25.3 |
| `delinq_2yrs` | Retards de paiement (2 ans) | 0, 1, 2 |
| `inq_last_6mths` | Demandes de crédit (6 mois) | 0, 1, 3 |
| `pub_rec` | Mentions légales publiques | 0, 1 |
| `total_acc` | Nombre total de comptes | 12, 25, 40 |

## 3.4 Prétraitement de la Variable Cible

```python
def pretraiter_cible(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible binaire good_bad.
    
    La variable cible indique si l'emprunteur est :
    - 1 (Good/Bon) : A remboursé son prêt
    - 0 (Bad/Mauvais) : En défaut de paiement
    
    Args:
        loan_data: DataFrame contenant les données de prêts
        
    Returns:
        DataFrame avec colonne 'good_bad' ajoutée
        
    Exemple:
        >>> df = pretraiter_cible(loan_data)
        >>> df['good_bad'].value_counts()
        1    750000  # Bons emprunteurs
        0     50000  # Mauvais emprunteurs
    """
    # Statuts considérés comme "mauvais" (défaut)
    statuts_mauvais = [
        'Charged Off',                                    # Passé en perte
        'Default',                                        # Défaut
        'Does not meet the credit policy. Status: Charged Off.',  # Hors politique
        'Late (31-120 days)'                              # Retard > 31 jours
    ]
    
    # 1 = Good (bon), 0 = Bad (mauvais)
    loan_data['good_bad'] = np.where(
        loan_data['loan_status'].isin(statuts_mauvais), 
        0,  # Bad
        1   # Good
    )
    
    return loan_data
```

## 3.5 Création des Variables Dummy (Indicatrices)

**Qu'est-ce qu'une variable dummy ?**
Une variable dummy (ou indicatrice) transforme une variable catégorielle en plusieurs colonnes binaires (0/1).

**Exemple** :
```
Grade original : A, B, C
Après transformation :
- grade:A = 1 si Grade=A, sinon 0
- grade:B = 1 si Grade=B, sinon 0
- grade:C = 1 si Grade=C, sinon 0
```

```python
def creer_variables_dummy(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables dummy pour les colonnes catégorielles.
    
    Le piège des variables dummy (dummy variable trap) :
    Si une variable a N catégories, on n'utilise que N-1 colonnes dummy
    dans le modèle pour éviter la multicolinéarité parfaite.
    
    Exemple avec la note (grade) :
        Pour 7 grades (A à G), on utilise 6 colonnes,
        la 7ème (G) devient la catégorie de référence.
    
    Args:
        loan_data: DataFrame avec les données brutes
        
    Returns:
        DataFrame avec colonnes dummy ajoutées
    """
    colonnes_dummy = [
        'grade',            # Note du prêt
        'sub_grade',        # Sous-note
        'home_ownership',   # Propriété
        'verification_status',  # Vérification
        'loan_status',      # Statut du prêt
        'purpose',          # Objet du prêt
        'addr_state',       # État
        'initial_list_status'  # Listing initial
    ]
    
    df_dummies = pd.DataFrame()
    
    for col in colonnes_dummy:
        # pd.get_dummies crée une colonne pour chaque catégorie unique
        # prefix='grade' et prefix_sep=':' donnent des noms comme 'grade:A'
        df_dummy = pd.get_dummies(
            loan_data[col], 
            prefix=col, 
            prefix_sep=':'
        )
        df_dummies = pd.concat([df_dummies, df_dummy], axis=1)
    
    # Fusion avec le dataset principal
    loan_data = pd.concat([loan_data, df_dummies], axis=1)
    
    return loan_data

# Application
loan_data = creer_variables_dummy(loan_data)
print(f"Colonnes dummy créées: {df_dummies.shape[1]}")
```

## 3.6 Conversion des Variables Continues

```python
def convertir_emp_length(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit emp_length (durée d'emploi) en entier.
    
    Transformations :
    - '10+ years' → 10
    - '< 1 year' → 0
    - 'n/a' → 0
    - '5 years' → 5
    
    Args:
        loan_data: DataFrame avec emp_length en string
        
    Returns:
        DataFrame avec emp_length_int en entier
    """
    col = 'emp_length_int'
    loan_data[col] = loan_data['emp_length'].str.replace('+ years', '')
    loan_data[col] = loan_data[col].str.replace('< 1 year', '0')
    loan_data[col] = loan_data[col].str.replace('n/a', '0')
    loan_data[col] = loan_data[col].str.replace(' years', '')
    loan_data[col] = loan_data[col].str.replace(' year', '')
    loan_data[col] = pd.to_numeric(loan_data[col])
    
    return loan_data

def convertir_dates(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les dates en nombre de mois depuis une date de référence.
    
    Exemple :
        'Apr-03' (Avril 2003) avec référence Décembre 2017
        → 176 mois depuis la première ligne de crédit
    
    Args:
        loan_data: DataFrame avec dates en format 'MMM-YY'
        
    Returns:
        DataFrame avec mths_since_earliest_cr_line en entier
    """
    # Date de référence (on suppose être en Décembre 2017)
    date_reference = pd.to_datetime('2017-12-01')
    
    # Conversion de la date de première ligne de crédit
    loan_data['earliest_cr_line_date'] = pd.to_datetime(
        loan_data['earliest_cr_line'], 
        format='%b-%y'
    )
    
    # Calcul de la différence en mois
    diff = date_reference - loan_data['earliest_cr_line_date']
    loan_data['mths_since_earliest_cr_line'] = round(
        pd.to_numeric(diff / np.timedelta64(1, 'M'))
    )
    
    return loan_data
```

---

# 4. Modèle PD - Probabilité de Défaut

## 4.1 Concept de WoE (Weight of Evidence, Poids de l'Évidence)

Le **WoE** mesure la force prédictive d'une variable en séparant les "bons" des "mauvais" emprunteurs.

**Formule** :
```
WoE = ln(% de Bons dans la catégorie / % de Mauvais dans la catégorie)
```

**Interprétation** :
- WoE > 0 : La catégorie a plus de "bons" emprunteurs que la moyenne
- WoE < 0 : La catégorie a plus de "mauvais" emprunteurs que la moyenne
- WoE = 0 : Distribution neutre

**Exemple concret** :
Pour la note (grade) A :
- 15% des "bons" emprunteurs ont un grade A
- 5% des "mauvais" emprunteurs ont un grade A
- WoE = ln(15/5) = ln(3) ≈ 1.10

Cela signifie que le grade A est associé à de meilleurs emprunteurs.

```python
def calculer_woe_discret(
    df: pd.DataFrame, 
    variable_name: str, 
    target_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule le WoE (Weight of Evidence, Poids de l'Évidence) 
    et l'IV (Information Value, Valeur d'Information) 
    pour une variable catégorielle.
    
    Le WoE mesure la capacité d'une variable à discriminer 
    entre les bons et mauvais emprunteurs.
    
    Args:
        df: DataFrame avec la variable explicative
        variable_name: Nom de la colonne à analyser
        target_df: DataFrame avec la variable cible (good_bad)
        
    Returns:
        DataFrame avec WoE et IV par catégorie
        
    Exemple:
        >>> woe_grade = calculer_woe_discret(
        ...     loan_data, 'grade', loan_data[['good_bad']]
        ... )
        >>> print(woe_grade[['grade', 'WoE', 'IV']])
        
           grade      WoE       IV
        0      A    1.102    0.156
        1      B    0.453    0.082
        2      C    0.121    0.015
        ...
    """
    # 1. Concaténation de la variable et de la cible
    df = pd.concat([df[variable_name], target_df], axis=1)
    
    # 2. Calcul du nombre d'observations et de la moyenne (proportion de "bons")
    df = pd.concat([
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()
    ], axis=1)
    
    # 3. Nettoyage et renommage
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    
    # 4. Proportion d'observations
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    
    # 5. Nombre de bons et mauvais
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    
    # 6. Proportions de bons et mauvais
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    
    # 7. Calcul du WoE
    # WoE = ln(Distribution des Bons / Distribution des Mauvais)
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    
    # 8. Tri par WoE
    df = df.sort_values(['WoE']).reset_index(drop=True)
    
    # 9. Différences pour analyse
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    
    # 10. Calcul de l'IV (Information Value)
    # IV = Σ (prop_good - prop_bad) × WoE
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    
    return df
```

## 4.2 IV (Information Value, Valeur d'Information)

L'**IV** mesure le pouvoir prédictif global d'une variable.

**Formule** :
```
IV = Σ (Distribution Bons - Distribution Mauvais) × WoE
```

**Règles d'interprétation** :

| IV | Pouvoir Prédictif |
|----|-------------------|
| < 0.02 | Inutile |
| 0.02 - 0.1 | Faible |
| 0.1 - 0.3 | Moyen |
| 0.3 - 0.5 | Fort |
| > 0.5 | Suspect (overfitting possible) |

```python
def visualiser_woe(df_woe: pd.DataFrame, rotation: int = 0) -> None:
    """
    Visualise le WoE par catégorie.
    
    Args:
        df_woe: DataFrame avec les calculs WoE
        rotation: Rotation des labels de l'axe x
    """
    x = np.array(df_woe.iloc[:, 0].apply(str))
    y = df_woe['WoE']
    
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of Evidence (WoE)')
    plt.title(f'WoE par {df_woe.columns[0]}')
    plt.xticks(rotation=rotation)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## 4.3 Fine Classing et Coarse Classing

### Fine Classing (Découpage Fin)

Le **fine classing** divise les variables continues en de nombreuses petites catégories pour analyser la relation avec la cible.

**Exemple** : Revenu annuel divisé en tranches de 10 000 €.

### Coarse Classing (Découpage Grossier)

Le **coarse classing** regroupe les catégories ayant des WoE similaires pour simplifier le modèle.

**Exemple** : Les tranches de revenu 60K-70K et 70K-80K avec WoE proches sont fusionnées.

```python
def creer_categories_revenu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les catégories de revenu annuel après coarse classing.
    
    Le découpage est basé sur l'analyse WoE qui montre 
    des comportements de remboursement différents par tranche.
    
    Args:
        df: DataFrame avec la colonne annual_inc
        
    Returns:
        DataFrame avec colonnes dummy pour le revenu
        
    Exemple:
        Après transformation, on obtient :
        - annual_inc:<20K = 1 si revenu < 20000, sinon 0
        - annual_inc:20K-30K = 1 si 20000 < revenu <= 30000, sinon 0
        - etc.
    """
    # Catégories basées sur l'analyse WoE
    df['annual_inc:<20K'] = np.where(df['annual_inc'] <= 20000, 1, 0)
    df['annual_inc:20K-30K'] = np.where(
        (df['annual_inc'] > 20000) & (df['annual_inc'] <= 30000), 1, 0
    )
    df['annual_inc:30K-40K'] = np.where(
        (df['annual_inc'] > 30000) & (df['annual_inc'] <= 40000), 1, 0
    )
    df['annual_inc:40K-50K'] = np.where(
        (df['annual_inc'] > 40000) & (df['annual_inc'] <= 50000), 1, 0
    )
    df['annual_inc:50K-60K'] = np.where(
        (df['annual_inc'] > 50000) & (df['annual_inc'] <= 60000), 1, 0
    )
    df['annual_inc:60K-70K'] = np.where(
        (df['annual_inc'] > 60000) & (df['annual_inc'] <= 70000), 1, 0
    )
    df['annual_inc:70K-80K'] = np.where(
        (df['annual_inc'] > 70000) & (df['annual_inc'] <= 80000), 1, 0
    )
    df['annual_inc:80K-90K'] = np.where(
        (df['annual_inc'] > 80000) & (df['annual_inc'] <= 90000), 1, 0
    )
    df['annual_inc:90K-100K'] = np.where(
        (df['annual_inc'] > 90000) & (df['annual_inc'] <= 100000), 1, 0
    )
    df['annual_inc:100K-120K'] = np.where(
        (df['annual_inc'] > 100000) & (df['annual_inc'] <= 120000), 1, 0
    )
    df['annual_inc:120K-140K'] = np.where(
        (df['annual_inc'] > 120000) & (df['annual_inc'] <= 140000), 1, 0
    )
    df['annual_inc:>140K'] = np.where(df['annual_inc'] > 140000, 1, 0)
    
    return df
```

## 4.4 Régression Logistique avec P-Values

Scikit-learn ne fournit pas les p-values par défaut. Cette classe personnalisée les calcule.

```python
class LogisticRegression_with_p_values:
    """
    Classe de régression logistique qui calcule les p-values 
    pour tester la significativité statistique des coefficients.
    
    Pourquoi les p-values sont importantes ?
    -----------------------------------------
    Une p-value < 0.05 indique que le coefficient est 
    statistiquement significatif, c'est-à-dire qu'il y a 
    moins de 5% de chances que l'effet observé soit dû au hasard.
    
    Comment ça marche ?
    -------------------
    1. On calcule la FIM (Fisher Information Matrix, 
       Matrice d'Information de Fisher)
    2. On inverse pour obtenir la matrice de Cramér-Rao
    3. La diagonale donne les variances des coefficients
    4. On calcule les z-scores et les p-values
    
    Attributs:
        model: Le modèle LogisticRegression de sklearn
        coef_: Les coefficients du modèle
        intercept_: L'ordonnée à l'origine
        p_values: Les p-values de chaque coefficient
        
    Exemple:
        >>> model = LogisticRegression_with_p_values()
        >>> model.fit(X_train, y_train)
        >>> print(f"P-values: {model.p_values}")
        >>> # Coefficients significatifs si p-value < 0.05
    """
    
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraîne le modèle et calcule les p-values.
        
        Args:
            X: Matrice des variables explicatives (n_samples, n_features)
            y: Vecteur cible (n_samples,)
        """
        # Entraînement du modèle sklearn
        self.model.fit(X, y)
        
        # ============================================
        # Calcul des p-values
        # ============================================
        
        # Dénominateur pour la FIM
        # decision_function renvoie les logits : w.T @ x + b
        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X)))
        denom = np.tile(denom, (X.shape[1], 1)).T
        
        # FIM (Fisher Information Matrix)
        # Mesure la quantité d'information sur les paramètres
        F_ij = np.dot((X / denom).T, X)
        
        # Matrice de Cramér-Rao (inverse de FIM)
        # Donne la borne inférieure de la variance des estimateurs
        Cramer_Rao = np.linalg.inv(F_ij)
        
        # Erreurs standards = racine carrée de la diagonale
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        
        # Z-scores = coefficient / erreur standard
        z_scores = self.model.coef_[0] / sigma_estimates
        
        # P-values (test bilatéral)
        # sf = survival function = 1 - CDF
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        
        # Stockage des résultats
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
        
        return self
```

## 4.5 Liste des Variables Sélectionnées

Après l'analyse WoE/IV, les variables suivantes sont retenues pour le modèle PD :

```python
# Variables utilisées dans le modèle PD
features_all = [
    # Grade (note du prêt) - Fort pouvoir prédictif
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    
    # Propriété du logement
    'home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:OWN', 'home_ownership:MORTGAGE',
    
    # État de résidence (regroupés par WoE similaire)
    'addr_state:ND_NE_IA_NV_FL_HI_AL', 'addr_state:NM_VA', 'addr_state:NY',
    'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA', 'addr_state:UT_KY_AZ_NJ',
    'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN', 'addr_state:GA_WA_OR',
    'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT',
    'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID',
    
    # Statut de vérification
    'verification_status:Not Verified', 'verification_status:Source Verified',
    'verification_status:Verified',
    
    # Objet du prêt (regroupés)
    'purpose:educ__sm_b__wedd__ren_en__mov__house', 'purpose:credit_card',
    'purpose:debt_consolidation', 'purpose:oth__med__vacation',
    'purpose:major_purch__car__home_impr',
    
    # Listing initial
    'initial_list_status:f', 'initial_list_status:w',
    
    # Durée du prêt
    'term:36', 'term:60',
    
    # Années d'emploi
    'emp_length:0', 'emp_length:1', 'emp_length:2-4', 
    'emp_length:5-6', 'emp_length:7-9', 'emp_length:10',
    
    # Taux d'intérêt (très prédictif)
    'int_rate:<9.548', 'int_rate:9.548-12.025', 'int_rate:12.025-15.74',
    'int_rate:15.74-20.281', 'int_rate:>20.281',
    
    # Demandes de crédit récentes
    'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'inq_last_6mths:>6',
    
    # Mentions publiques
    'pub_rec:0-2', 'pub_rec:3-4', 'pub_rec:>=5',
    
    # Comptes en retard
    'acc_now_delinq:0', 'acc_now_delinq:>=1',
    
    # Limite de crédit revolving
    'total_rev_hi_lim:<=5K', 'total_rev_hi_lim:5K-10K', 'total_rev_hi_lim:10K-20K',
    'total_rev_hi_lim:20K-30K', 'total_rev_hi_lim:30K-40K', 'total_rev_hi_lim:40K-55K',
    'total_rev_hi_lim:55K-95K', 'total_rev_hi_lim:>95K',
    
    # Revenu annuel
    'annual_inc:<20K', 'annual_inc:20K-30K', 'annual_inc:30K-40K',
    'annual_inc:40K-50K', 'annual_inc:50K-60K', 'annual_inc:60K-70K',
    'annual_inc:70K-80K', 'annual_inc:80K-90K', 'annual_inc:90K-100K',
    'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K',
    
    # DTI (Debt-to-Income ratio)
    'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5',
    'dti:10.5-16.1', 'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4',
    'dti:22.4-35', 'dti:>35',
]

# Catégories de référence (à exclure pour éviter le piège des dummy)
ref_categories = [
    'grade:G',                          # Pire note = référence
    'home_ownership:RENT_OTHER_NONE_ANY',
    'addr_state:ND_NE_IA_NV_FL_HI_AL',
    'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house',
    'initial_list_status:f',
    'term:60',                          # Prêt long = référence
    'emp_length:0',                     # Sans emploi = référence
    'int_rate:>20.281',                 # Taux élevé = référence
    'inq_last_6mths:>6',
    'pub_rec:0-2',
    'acc_now_delinq:0',
    'total_rev_hi_lim:<=5K',
    'annual_inc:<20K',                  # Faible revenu = référence
    'dti:>35',                          # DTI élevé = référence
]
```

## 4.6 Entraînement du Modèle PD

```python
def entrainer_modele_pd(
    inputs_train: pd.DataFrame, 
    targets_train: pd.DataFrame
) -> LogisticRegression_with_p_values:
    """
    Entraîne le modèle PD (Probability of Default) par régression logistique.
    
    Args:
        inputs_train: Variables explicatives d'entraînement
        targets_train: Variable cible (good_bad)
        
    Returns:
        Modèle entraîné avec p-values
        
    Exemple:
        >>> model = entrainer_modele_pd(X_train, y_train)
        >>> print(f"Intercept: {model.intercept_[0]:.4f}")
        >>> print(f"Nombre de coefficients: {len(model.coef_[0])}")
    """
    # Sélection des variables
    X_train = inputs_train[features_all].copy()
    
    # Suppression des catégories de référence
    X_train = X_train.drop(ref_categories, axis=1)
    
    # Entraînement
    model = LogisticRegression_with_p_values()
    model.fit(X_train, targets_train)
    
    # Résumé des résultats
    feature_names = X_train.columns.values
    summary = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'p_value': model.p_values
    })
    
    # Ajout de l'intercept
    intercept_row = pd.DataFrame({
        'Feature': ['Intercept'],
        'Coefficient': [model.intercept_[0]],
        'p_value': [np.nan]
    })
    summary = pd.concat([intercept_row, summary], ignore_index=True)
    
    print("\n=== Résumé du Modèle PD ===")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Variables significatives (p < 0.05): "
          f"{sum(np.array(model.p_values) < 0.05)}/{len(model.p_values)}")
    
    return model, summary

# Entraînement
model_pd, summary_pd = entrainer_modele_pd(inputs_train, targets_train)

# Sauvegarde du modèle
import pickle
pickle.dump(model_pd, open('models/pd_model.sav', 'wb'))
```

---

# 5. Scorecard et Seuils de Décision

## 5.1 Concept de Scorecard (Carte de Score)

Une **scorecard** convertit les coefficients de la régression logistique en scores faciles à interpréter.

**Pourquoi une scorecard ?**
- Exigence réglementaire (Bâle II)
- Facilité d'explication aux clients et régulateurs
- Standardisation (ex: échelle FICO de 300 à 850)

## 5.2 Calcul des Scores

**Formule de base** :
```
Score = (β₀ + β₁x₁ + β₂x₂ + ...) × Factor + Offset
```

Où :
- β₀ = intercept
- βᵢ = coefficients
- Factor et Offset ajustent l'échelle

**Paramètres standards** :
- Score minimum : 300
- Score maximum : 850
- PDO (Points to Double Odds) : 20 points

```python
def creer_scorecard(
    summary: pd.DataFrame,
    min_score: int = 300,
    max_score: int = 850
) -> pd.DataFrame:
    """
    Crée une scorecard à partir des coefficients du modèle PD.
    
    La scorecard attribue un score à chaque catégorie de variable
    de façon à ce que le score total reflète la probabilité de défaut.
    
    Paramètres de l'échelle FICO :
    - Score minimum : 300 (très mauvais crédit)
    - Score maximum : 850 (excellent crédit)
    
    Args:
        summary: DataFrame avec Feature, Coefficient, p_value
        min_score: Score minimum de l'échelle
        max_score: Score maximum de l'échelle
        
    Returns:
        DataFrame scorecard avec les scores par catégorie
        
    Exemple:
        Si le coefficient de 'grade:A' est 0.166 et le factor est 76.7,
        alors le score pour grade:A ≈ 0.166 × 76.7 ≈ 13 points
    """
    # Copie du résumé
    scorecard = summary.copy()
    
    # Extraction de la variable d'origine
    scorecard['Original_Feature'] = scorecard['Feature'].apply(
        lambda x: x.split(':')[0] if ':' in x else x
    )
    
    # Calcul du factor (échelle)
    # Le factor convertit les log-odds en scores
    # On utilise la formule : factor = PDO / ln(2)
    pdo = 20  # Points to Double Odds
    factor = pdo / np.log(2)  # ≈ 28.85
    
    # Calcul des scores préliminaires
    # Score = Coefficient × Factor
    scorecard['Score_Calculation'] = scorecard['Coefficient'] * factor
    
    # Ajustement pour l'intercept
    # On répartit l'intercept sur toutes les variables
    n_variables = scorecard['Original_Feature'].nunique() - 1  # -1 pour l'intercept
    intercept_contribution = scorecard.loc[
        scorecard['Feature'] == 'Intercept', 'Score_Calculation'
    ].values[0] / n_variables
    
    # Score préliminaire avec distribution de l'intercept
    scorecard['Score_Preliminary'] = scorecard['Score_Calculation'].apply(
        lambda x: round(x + intercept_contribution) if x != 0 else 0
    )
    
    # Normalisation sur l'échelle souhaitée
    # Score final entre min_score et max_score
    current_min = scorecard['Score_Preliminary'].min()
    current_max = scorecard['Score_Preliminary'].max()
    
    scorecard['Score_Final'] = scorecard['Score_Preliminary'].apply(
        lambda x: round(
            min_score + (x - current_min) * (max_score - min_score) / 
            (current_max - current_min)
        ) if current_max != current_min else min_score
    )
    
    return scorecard

# Création de la scorecard
scorecard = creer_scorecard(summary_pd)

# Sauvegarde
scorecard.to_csv('data/df_scorecard.csv')
print("\n=== Aperçu de la Scorecard ===")
print(scorecard[['Feature', 'Coefficient', 'Score_Final']].head(15))
```

## 5.3 Exemple de Scorecard Réelle

| Variable | Catégorie | Score |
|----------|-----------|-------|
| **Intercept** | - | 442 |
| **grade** | A | +13 |
| **grade** | B | +24 |
| **grade** | C | +24 |
| **grade** | D | +22 |
| **grade** | E | +19 |
| **grade** | F | +7 |
| **grade** | G (référence) | 0 |
| **int_rate** | < 9.548% | +127 |
| **int_rate** | 9.548-12.025% | +77 |
| **int_rate** | 12.025-15.74% | +50 |
| **int_rate** | > 20.281% (référence) | 0 |
| **annual_inc** | > 140K | +35 |
| **annual_inc** | 100K-120K | +33 |
| **annual_inc** | < 20K (référence) | 0 |

**Exemple de calcul** :
Un client avec Grade B, taux 11%, revenu 80K :
```
Score = 442 + 24 (grade B) + 77 (int_rate) + 25 (annual_inc 80-90K) + ...
```

## 5.4 Seuils de Décision (Cutoffs)

Le **seuil de décision** détermine quels clients obtiennent un crédit.

```python
def analyser_seuils(
    df_predictions: pd.DataFrame,
    seuils: list = [0.8, 0.85, 0.9, 0.95]
) -> pd.DataFrame:
    """
    Analyse l'impact de différents seuils de décision.
    
    Args:
        df_predictions: DataFrame avec colonnes 'target' et 'proba'
        seuils: Liste des seuils à tester
        
    Returns:
        DataFrame avec métriques par seuil
        
    Exemple:
        Un seuil de 0.90 signifie :
        - Si P(Good) > 90% → Accepter le crédit
        - Si P(Good) ≤ 90% → Refuser le crédit
    """
    resultats = []
    
    for seuil in seuils:
        # Prédiction binaire basée sur le seuil
        predictions = np.where(df_predictions['proba'] > seuil, 1, 0)
        actuel = df_predictions['target']
        
        # Calcul des métriques
        tp = sum((predictions == 1) & (actuel == 1))  # Vrais positifs
        tn = sum((predictions == 0) & (actuel == 0))  # Vrais négatifs
        fp = sum((predictions == 1) & (actuel == 0))  # Faux positifs
        fn = sum((predictions == 0) & (actuel == 1))  # Faux négatifs
        
        total = len(actuel)
        accuracy = (tp + tn) / total
        taux_acceptation = sum(predictions == 1) / total
        taux_defaut_acceptes = fp / (tp + fp) if (tp + fp) > 0 else 0
        
        resultats.append({
            'Seuil': seuil,
            'Accuracy': f"{accuracy:.2%}",
            'Taux_Acceptation': f"{taux_acceptation:.2%}",
            'Taux_Defaut_Acceptes': f"{taux_defaut_acceptes:.2%}",
            'Vrais_Positifs': tp,
            'Faux_Positifs': fp
        })
    
    return pd.DataFrame(resultats)

# Exemple d'utilisation
print("\n=== Analyse des Seuils de Décision ===")
print("""
| Seuil | Acceptation | Défauts Acceptés | Interprétation |
|-------|-------------|------------------|----------------|
| 0.80  | 75%         | 8%               | Politique libérale |
| 0.85  | 60%         | 5%               | Équilibré |
| 0.90  | 45%         | 3%               | Conservateur |
| 0.95  | 25%         | 1%               | Très restrictif |
""")
```

---

# 6. Modèle LGD - Perte en Cas de Défaut

## 6.1 Concept de LGD

La **LGD (Loss Given Default, Perte en Cas de Défaut)** mesure le pourcentage du montant exposé qui est perdu en cas de défaut.

**Formule** :
```
LGD = 1 - Taux de Récupération
```

**Exemple** :
- Prêt en défaut : 10 000 €
- Montant récupéré (saisie, vente) : 7 000 €
- Taux de récupération : 70%
- LGD : 30%

## 6.2 Modèle en Deux Étapes

Le modèle LGD utilise une approche en deux étapes car la distribution du taux de récupération est bimodale (beaucoup de 0% et 100%).

### Étape 1 : Classification (Récupération ou Non)

```python
def entrainer_lgd_etape1(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> LogisticRegression_with_p_values:
    """
    Étape 1 du modèle LGD : Classification binaire.
    
    Prédit si le taux de récupération est > 0 ou = 0.
    
    Variable cible :
    - 1 : Récupération > 0 (on récupère quelque chose)
    - 0 : Récupération = 0 (perte totale)
    
    Args:
        X_train: Variables explicatives
        y_train: recovery_rate_0_1 (1 si recovery > 0, 0 sinon)
        
    Returns:
        Modèle de régression logistique entraîné
        
    Exemple:
        >>> model_lgd_1 = entrainer_lgd_etape1(X, y)
        >>> # Prédit la probabilité de récupérer quelque chose
        >>> proba_recovery = model_lgd_1.predict_proba(X_test)[:, 1]
    """
    # Création de la variable cible binaire
    # recovery_rate_0_1 = 1 si recovery_rate > 0, sinon 0
    
    # Variables pour le modèle LGD
    features_lgd = [
        'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F',
        'home_ownership:MORTGAGE', 'home_ownership:OWN', 'home_ownership:RENT',
        'verification_status:Not Verified', 'verification_status:Source Verified',
        'purpose:car', 'purpose:credit_card', 'purpose:debt_consolidation',
        'purpose:home_improvement', 'purpose:major_purchase', 'purpose:medical',
        'purpose:other', 'purpose:small_business',
        'initial_list_status:w',
        'term_int', 'int_rate', 'funded_amnt', 'annual_inc', 'dti',
        'mths_since_last_delinq', 'mths_since_last_record'
    ]
    
    # Catégories de référence
    ref_cat_lgd = [
        'grade:G', 'home_ownership:OTHER', 'verification_status:Verified',
        'purpose:educational', 'initial_list_status:f'
    ]
    
    X = X_train[features_lgd].drop(ref_cat_lgd, axis=1, errors='ignore')
    
    model = LogisticRegression_with_p_values()
    model.fit(X, y_train)
    
    return model

# Entraînement Étape 1
model_lgd_step1 = entrainer_lgd_etape1(lgd_X_train, lgd_y_train_step1)

# Sauvegarde
pickle.dump(model_lgd_step1, open('models/lgd_model_step_1.sav', 'wb'))
```

### Étape 2 : Régression (Montant de Récupération)

```python
class LinearRegression_with_p_values(linear_model.LinearRegression):
    """
    Régression linéaire avec calcul des p-values.
    
    Utilisée pour l'étape 2 du modèle LGD qui prédit
    le taux de récupération exact (entre 0 et 1).
    
    Calcul des p-values :
    1. SSE (Sum of Squared Errors) = Σ(y - ŷ)²
    2. SE (Standard Error) = √(SSE / (n - p) × (X'X)⁻¹)
    3. t-statistic = coefficient / SE
    4. p-value = 2 × P(T > |t|)
    
    Attributs:
        t: t-statistics pour chaque coefficient
        p: p-values pour chaque coefficient
    """
    
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=1):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs)
    
    def fit(self, X, y, n_jobs=1):
        self = super().fit(X, y)
        
        # Calcul de SSE (Sum of Squared Errors)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0)
        sse = sse / float(X.shape[0] - X.shape[1])
        
        # Calcul de SE (Standard Error)
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        
        # t-statistic
        self.t = self.coef_ / se
        
        # p-values (distribution t de Student)
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        
        return self

def entrainer_lgd_etape2(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> LinearRegression_with_p_values:
    """
    Étape 2 du modèle LGD : Régression du taux de récupération.
    
    Ne s'applique qu'aux observations où recovery_rate > 0.
    Prédit le montant exact du taux de récupération.
    
    Args:
        X_train: Variables explicatives (cas avec récupération > 0)
        y_train: Taux de récupération réel (entre 0 et 1)
        
    Returns:
        Modèle de régression linéaire entraîné
        
    Exemple:
        >>> model_lgd_2 = entrainer_lgd_etape2(X_recovery, y_recovery)
        >>> # Prédit le taux de récupération
        >>> recovery_rate = model_lgd_2.predict(X_test)
        >>> recovery_rate = np.clip(recovery_rate, 0, 1)  # Borne [0, 1]
    """
    model = LinearRegression_with_p_values()
    model.fit(X_train, y_train)
    
    return model
```

## 6.3 Combinaison des Deux Étapes

```python
def predire_lgd(
    X: pd.DataFrame,
    model_step1: LogisticRegression_with_p_values,
    model_step2: LinearRegression_with_p_values
) -> np.ndarray:
    """
    Prédit la LGD en combinant les deux étapes.
    
    Formule :
    LGD = 1 - (P(recovery > 0) × E[recovery | recovery > 0])
    
    Où :
    - P(recovery > 0) vient de l'étape 1 (logistique)
    - E[recovery | recovery > 0] vient de l'étape 2 (linéaire)
    
    Args:
        X: Variables explicatives
        model_step1: Modèle de classification
        model_step2: Modèle de régression
        
    Returns:
        Array des LGD prédites (entre 0 et 1)
        
    Exemple:
        >>> lgd = predire_lgd(X_test, model_lgd_1, model_lgd_2)
        >>> print(f"LGD moyenne: {lgd.mean():.2%}")
        >>> # LGD moyenne: 42% signifie qu'on perd en moyenne 42% en cas de défaut
    """
    # Étape 1 : Probabilité de récupération > 0
    proba_recovery = model_step1.model.predict_proba(X)[:, 1]
    
    # Étape 2 : Taux de récupération prédit (si recovery > 0)
    recovery_rate = model_step2.predict(X)
    
    # Bornage entre 0 et 1
    recovery_rate = np.clip(recovery_rate, 0, 1)
    
    # Combinaison : E[recovery] = P(recovery > 0) × E[recovery | recovery > 0]
    expected_recovery = proba_recovery * recovery_rate
    
    # LGD = 1 - taux de récupération
    lgd = 1 - expected_recovery
    
    return lgd
```

---

# 7. Modèle EAD - Exposition au Défaut

## 7.1 Concept de EAD

L'**EAD (Exposure at Default, Exposition au Défaut)** est le montant que l'emprunteur doit au moment du défaut.

**Pour les prêts amortissables** :
```
EAD = Montant Financé - Principal Remboursé
```

## 7.2 CCF (Credit Conversion Factor, Facteur de Conversion du Crédit)

Le **CCF** mesure la portion du crédit utilisée au moment du défaut.

```python
def calculer_ccf(loan_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le CCF (Credit Conversion Factor).
    
    CCF = (Montant Financé - Principal Remboursé) / Montant Financé
    
    Interprétation :
    - CCF = 1 : Rien n'a été remboursé
    - CCF = 0.5 : La moitié a été remboursée
    - CCF = 0 : Tout a été remboursé (rare pour un défaut)
    
    Args:
        loan_data: DataFrame avec funded_amnt et total_rec_prncp
        
    Returns:
        DataFrame avec colonne CCF ajoutée
        
    Exemple:
        >>> loan_data = calculer_ccf(loan_data)
        >>> print(loan_data['CCF'].describe())
        count    50000
        mean     0.65   # En moyenne, 65% du prêt reste dû au défaut
        std      0.25
        min      0.01
        max      1.00
    """
    # CCF = (montant financé - principal remboursé) / montant financé
    loan_data['CCF'] = (
        loan_data['funded_amnt'] - loan_data['total_rec_prncp']
    ) / loan_data['funded_amnt']
    
    # Bornage entre 0 et 1
    loan_data['CCF'] = np.clip(loan_data['CCF'], 0, 1)
    
    return loan_data

def entrainer_modele_ead(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> LinearRegression_with_p_values:
    """
    Entraîne le modèle EAD par régression linéaire sur le CCF.
    
    Le modèle prédit le CCF, puis :
    EAD = CCF × Limite de Crédit Originale
    
    Args:
        X_train: Variables explicatives
        y_train: CCF (Credit Conversion Factor)
        
    Returns:
        Modèle de régression linéaire
        
    Performance attendue :
    - R² ≈ 0.65-0.70
    """
    model = LinearRegression_with_p_values()
    model.fit(X_train, y_train)
    
    print(f"R² du modèle EAD : {model.score(X_train, y_train):.3f}")
    
    return model
```

---

# 8. Calcul de la Perte Attendue (EL)

## 8.1 Formule Finale

```python
def calculer_expected_loss(
    pd_predictions: np.ndarray,
    lgd_predictions: np.ndarray,
    ead_predictions: np.ndarray
) -> np.ndarray:
    """
    Calcule la perte attendue (Expected Loss) pour chaque prêt.
    
    Formule :
    EL = PD × LGD × EAD
    
    Où :
    - PD (Probability of Default) : Probabilité de défaut (0-1)
    - LGD (Loss Given Default) : Pourcentage perdu en cas de défaut (0-1)
    - EAD (Exposure at Default) : Montant exposé en €
    
    Args:
        pd_predictions: Probabilités de défaut par prêt
        lgd_predictions: LGD par prêt
        ead_predictions: EAD en € par prêt
        
    Returns:
        Array des pertes attendues en €
        
    Exemple numérique :
        Client A :
        - PD = 5% (0.05)
        - LGD = 40% (0.40)
        - EAD = 15 000 €
        
        EL = 0.05 × 0.40 × 15 000 = 300 €
        
        La banque doit provisionner 300 € pour ce client.
    """
    expected_loss = pd_predictions * lgd_predictions * ead_predictions
    
    return expected_loss

def calculer_el_portefeuille(
    loan_data: pd.DataFrame,
    model_pd,
    model_lgd_step1,
    model_lgd_step2,
    model_ead
) -> dict:
    """
    Calcule la perte attendue pour tout le portefeuille.
    
    Args:
        loan_data: Données du portefeuille
        model_*: Modèles entraînés
        
    Returns:
        Dictionnaire avec les statistiques de perte
        
    Exemple de résultat :
        {
            'EL_Total': 15_000_000,      # Perte totale attendue
            'EL_Moyenne': 300,            # Perte moyenne par prêt
            'Taux_EL': 0.015,             # EL / Exposition totale
            'Nb_Prets': 50_000
        }
    """
    # Préparation des features
    X = preparer_features(loan_data)
    
    # Prédictions PD
    pd_pred = model_pd.model.predict_proba(X)[:, 0]  # Proba de défaut
    
    # Prédictions LGD
    lgd_pred = predire_lgd(X, model_lgd_step1, model_lgd_step2)
    
    # Prédictions EAD
    ccf_pred = model_ead.predict(X)
    ccf_pred = np.clip(ccf_pred, 0, 1)
    ead_pred = ccf_pred * loan_data['funded_amnt'].values
    
    # Calcul EL
    el = calculer_expected_loss(pd_pred, lgd_pred, ead_pred)
    
    # Statistiques
    resultats = {
        'EL_Total': el.sum(),
        'EL_Moyenne': el.mean(),
        'EL_Mediane': np.median(el),
        'EL_Std': el.std(),
        'Taux_EL': el.sum() / ead_pred.sum(),
        'Exposition_Totale': ead_pred.sum(),
        'Nb_Prets': len(el)
    }
    
    return resultats

# Exemple d'utilisation
print("\n=== Perte Attendue du Portefeuille ===")
resultats_el = calculer_el_portefeuille(loan_data, model_pd, model_lgd_1, model_lgd_2, model_ead)
print(f"EL Total : {resultats_el['EL_Total']:,.0f} €")
print(f"EL Moyenne par prêt : {resultats_el['EL_Moyenne']:,.0f} €")
print(f"Taux EL : {resultats_el['Taux_EL']:.2%}")
```

---

# 9. PSI - Indice de Stabilité de Population

## 9.1 Concept de PSI

Le **PSI (Population Stability Index, Indice de Stabilité de Population)** mesure si la population actuelle diffère significativement de la population d'entraînement.

**Pourquoi le PSI est important ?**
- Vérifie que le modèle reste valide dans le temps
- Détecte les dérives de population (population drift)
- Exigence réglementaire pour le suivi des modèles

## 9.2 Formule du PSI

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Où :
- Actual% = Distribution actuelle
- Expected% = Distribution d'entraînement

**Interprétation** :

| PSI | Signification | Action |
|-----|---------------|--------|
| < 0.10 | Pas de changement significatif | Continuer |
| 0.10 - 0.25 | Changement mineur | Surveiller |
| > 0.25 | Changement majeur | Recalibrer le modèle |

```python
def calculer_psi(
    expected: np.ndarray, 
    actual: np.ndarray, 
    n_bins: int = 10
) -> float:
    """
    Calcule le PSI (Population Stability Index).
    
    Le PSI compare la distribution des scores entre deux populations :
    - Expected : Population d'entraînement (2007-2014)
    - Actual : Population actuelle (2015)
    
    Args:
        expected: Scores de la population de référence
        actual: Scores de la population actuelle
        n_bins: Nombre de classes pour la discrétisation
        
    Returns:
        Valeur du PSI
        
    Exemple:
        >>> psi = calculer_psi(scores_train, scores_2015)
        >>> print(f"PSI = {psi:.3f}")
        >>> if psi < 0.10:
        ...     print("✓ Population stable")
        >>> elif psi < 0.25:
        ...     print("⚠ Changement mineur")
        >>> else:
        ...     print("✗ Recalibration nécessaire")
    """
    # Création des bins basés sur la population expected
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calcul des fréquences
    expected_freq, _ = np.histogram(expected, bins=bins)
    actual_freq, _ = np.histogram(actual, bins=bins)
    
    # Conversion en pourcentages
    expected_pct = expected_freq / expected_freq.sum()
    actual_pct = actual_freq / actual_freq.sum()
    
    # Remplacement des zéros pour éviter log(0)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Calcul du PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi

def rapport_psi_detaille(
    expected: np.ndarray, 
    actual: np.ndarray, 
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Génère un rapport PSI détaillé par bin.
    
    Args:
        expected: Scores de référence
        actual: Scores actuels
        n_bins: Nombre de bins
        
    Returns:
        DataFrame avec PSI par bin
        
    Exemple de sortie :
        | Bin | Expected% | Actual% | PSI_Contribution |
        |-----|-----------|---------|------------------|
        | 1   | 10.0%     | 8.5%    | 0.023           |
        | 2   | 10.0%     | 12.0%   | 0.038           |
        | ... | ...       | ...     | ...             |
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    expected_freq, _ = np.histogram(expected, bins=bins)
    actual_freq, _ = np.histogram(actual, bins=bins)
    
    expected_pct = expected_freq / expected_freq.sum()
    actual_pct = actual_freq / actual_freq.sum()
    
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_contribution = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    
    rapport = pd.DataFrame({
        'Bin': range(1, n_bins + 1),
        'Bin_Range': [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(n_bins)],
        'Expected_Pct': expected_pct,
        'Actual_Pct': actual_pct,
        'PSI_Contribution': psi_contribution
    })
    
    rapport['Expected_Pct'] = rapport['Expected_Pct'].apply(lambda x: f"{x:.1%}")
    rapport['Actual_Pct'] = rapport['Actual_Pct'].apply(lambda x: f"{x:.1%}")
    
    total_psi = psi_contribution.sum()
    print(f"\n=== PSI Total : {total_psi:.4f} ===")
    
    if total_psi < 0.10:
        print("✓ Population stable - Pas d'action requise")
    elif total_psi < 0.25:
        print("⚠ Changement mineur - Surveillance recommandée")
    else:
        print("✗ Changement majeur - Recalibration du modèle nécessaire")
    
    return rapport

# Exemple d'utilisation
print("\n=== Vérification de la Stabilité du Modèle ===")
psi = calculer_psi(scores_train, scores_2015)
rapport = rapport_psi_detaille(scores_train, scores_2015)
print(rapport)
```

---

# 10. Code Complet et Fonctions Utilitaires

## 10.1 Métriques d'Évaluation

```python
def evaluer_modele_classification(
    y_true: np.ndarray, 
    y_proba: np.ndarray,
    seuil: float = 0.5
) -> dict:
    """
    Évalue un modèle de classification avec plusieurs métriques.
    
    Métriques calculées :
    - Accuracy : (TP + TN) / Total
    - AUROC (Area Under ROC) : Capacité discriminante
    - Gini : 2 × AUROC - 1
    - KS (Kolmogorov-Smirnov) : Max(cum_bad - cum_good)
    
    Args:
        y_true: Valeurs réelles (0/1)
        y_proba: Probabilités prédites
        seuil: Seuil de classification
        
    Returns:
        Dictionnaire avec toutes les métriques
        
    Exemple:
        >>> metrics = evaluer_modele_classification(y_test, probas)
        >>> print(f"AUROC: {metrics['AUROC']:.3f}")
        >>> print(f"Gini: {metrics['Gini']:.3f}")
    """
    # Prédictions binaires
    y_pred = np.where(y_proba > seuil, 1, 0)
    
    # Matrice de confusion
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Accuracy
    accuracy = (tp + tn) / len(y_true)
    
    # AUROC (Area Under ROC)
    auroc = roc_auc_score(y_true, y_proba)
    
    # Gini
    gini = 2 * auroc - 1
    
    # KS (Kolmogorov-Smirnov)
    df = pd.DataFrame({'proba': y_proba, 'target': y_true})
    df = df.sort_values('proba')
    df['cum_good'] = (df['target'] == 1).cumsum() / (df['target'] == 1).sum()
    df['cum_bad'] = (df['target'] == 0).cumsum() / (df['target'] == 0).sum()
    ks = (df['cum_bad'] - df['cum_good']).max()
    
    return {
        'Accuracy': accuracy,
        'AUROC': auroc,
        'Gini': gini,
        'KS': ks,
        'True_Positives': tp,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn
    }

def tracer_courbe_roc(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Trace la courbe ROC (Receiver Operating Characteristic).
    
    La courbe ROC visualise le compromis entre :
    - TPR (True Positive Rate, Sensibilité) : TP / (TP + FN)
    - FPR (False Positive Rate) : FP / (FP + TN)
    
    Args:
        y_true: Valeurs réelles
        y_proba: Probabilités prédites
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    plt.fill_between(fpr, tpr, alpha=0.3)
    
    plt.xlabel('FPR (False Positive Rate, Taux de Faux Positifs)')
    plt.ylabel('TPR (True Positive Rate, Sensibilité)')
    plt.title('Courbe ROC (Receiver Operating Characteristic)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## 10.2 Pipeline Complet

```python
def pipeline_credit_risk(
    loan_data_path: str,
    output_dir: str = 'models/'
) -> dict:
    """
    Pipeline complet de modélisation du risque de crédit.
    
    Étapes :
    1. Chargement et prétraitement des données
    2. Entraînement du modèle PD (Probability of Default)
    3. Création de la scorecard
    4. Entraînement du modèle LGD (Loss Given Default)
    5. Entraînement du modèle EAD (Exposure at Default)
    6. Calcul de la perte attendue (Expected Loss)
    
    Args:
        loan_data_path: Chemin vers les données
        output_dir: Répertoire de sauvegarde des modèles
        
    Returns:
        Dictionnaire avec tous les modèles et métriques
        
    Exemple:
        >>> results = pipeline_credit_risk('data/loan_data.csv')
        >>> print(f"EL Total: {results['EL_Total']:,.0f} €")
    """
    print("=" * 60)
    print("PIPELINE DE MODÉLISATION DU RISQUE DE CRÉDIT")
    print("=" * 60)
    
    # ============================================
    # ÉTAPE 1 : Chargement et prétraitement
    # ============================================
    print("\n[1/6] Chargement des données...")
    loan_data = pd.read_csv(loan_data_path)
    print(f"  → {len(loan_data):,} observations chargées")
    
    print("\n[1/6] Prétraitement...")
    loan_data = pretraiter_cible(loan_data)
    loan_data = creer_variables_dummy(loan_data)
    loan_data = convertir_emp_length(loan_data)
    loan_data = convertir_dates(loan_data)
    
    # Split train/test
    train_data, test_data = train_test_split(
        loan_data, test_size=0.2, random_state=42
    )
    print(f"  → Train: {len(train_data):,} | Test: {len(test_data):,}")
    
    # ============================================
    # ÉTAPE 2 : Modèle PD
    # ============================================
    print("\n[2/6] Entraînement du modèle PD...")
    model_pd, summary_pd = entrainer_modele_pd(
        train_data[features_all], 
        train_data['good_bad']
    )
    
    # Évaluation
    pd_proba = model_pd.model.predict_proba(
        test_data[features_all].drop(ref_categories, axis=1)
    )[:, 1]
    metrics_pd = evaluer_modele_classification(test_data['good_bad'], pd_proba)
    print(f"  → AUROC: {metrics_pd['AUROC']:.3f}")
    print(f"  → Gini: {metrics_pd['Gini']:.3f}")
    print(f"  → KS: {metrics_pd['KS']:.3f}")
    
    # ============================================
    # ÉTAPE 3 : Scorecard
    # ============================================
    print("\n[3/6] Création de la scorecard...")
    scorecard = creer_scorecard(summary_pd)
    scorecard.to_csv(f'{output_dir}/scorecard.csv')
    print(f"  → Scorecard sauvegardée ({len(scorecard)} lignes)")
    
    # ============================================
    # ÉTAPE 4 : Modèle LGD
    # ============================================
    print("\n[4/6] Entraînement du modèle LGD...")
    
    # Filtrer les défauts
    defaults = loan_data[loan_data['good_bad'] == 0].copy()
    defaults = calculer_ccf(defaults)
    defaults['recovery_rate_0_1'] = np.where(defaults['recovery_rate'] > 0, 1, 0)
    
    # Étape 1 : Classification
    model_lgd_1 = entrainer_lgd_etape1(defaults, defaults['recovery_rate_0_1'])
    
    # Étape 2 : Régression (sur les cas avec recovery > 0)
    recovery_positive = defaults[defaults['recovery_rate'] > 0]
    model_lgd_2 = entrainer_lgd_etape2(recovery_positive, recovery_positive['recovery_rate'])
    
    print(f"  → Modèle LGD Étape 1 (Classification) entraîné")
    print(f"  → Modèle LGD Étape 2 (Régression) entraîné")
    
    # ============================================
    # ÉTAPE 5 : Modèle EAD
    # ============================================
    print("\n[5/6] Entraînement du modèle EAD...")
    model_ead = entrainer_modele_ead(defaults, defaults['CCF'])
    print(f"  → R² du modèle EAD: {model_ead.score(defaults, defaults['CCF']):.3f}")
    
    # ============================================
    # ÉTAPE 6 : Perte Attendue
    # ============================================
    print("\n[6/6] Calcul de la perte attendue...")
    resultats_el = calculer_el_portefeuille(
        test_data, model_pd, model_lgd_1, model_lgd_2, model_ead
    )
    print(f"  → EL Total: {resultats_el['EL_Total']:,.0f} €")
    print(f"  → EL Moyenne: {resultats_el['EL_Moyenne']:,.0f} €")
    print(f"  → Taux EL: {resultats_el['Taux_EL']:.2%}")
    
    # ============================================
    # Sauvegarde des modèles
    # ============================================
    print("\n[Sauvegarde des modèles...]")
    pickle.dump(model_pd, open(f'{output_dir}/pd_model.sav', 'wb'))
    pickle.dump(model_lgd_1, open(f'{output_dir}/lgd_model_step_1.sav', 'wb'))
    pickle.dump(model_lgd_2, open(f'{output_dir}/lgd_model_step_2.sav', 'wb'))
    pickle.dump(model_ead, open(f'{output_dir}/ead_model.sav', 'wb'))
    print(f"  → Modèles sauvegardés dans {output_dir}")
    
    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    
    return {
        'model_pd': model_pd,
        'model_lgd_1': model_lgd_1,
        'model_lgd_2': model_lgd_2,
        'model_ead': model_ead,
        'scorecard': scorecard,
        'metrics_pd': metrics_pd,
        'resultats_el': resultats_el
    }

# ============================================================
# EXÉCUTION DU PIPELINE
# ============================================================
if __name__ == "__main__":
    results = pipeline_credit_risk('data/loan_data_2007_2014.csv')
```

---

# Annexes

## A.1 Résumé des Performances des Modèles

| Modèle | Type | Métrique | Valeur |
|--------|------|----------|--------|
| **PD** | Régression Logistique | Accuracy | 57.2% |
| **PD** | Régression Logistique | AUROC | 0.684 |
| **PD** | Régression Logistique | Gini | 0.368 |
| **LGD Étape 1** | Régression Logistique | AUROC | 0.640 |
| **LGD Étape 2** | Régression Linéaire | R² | 0.777 |
| **EAD** | Régression Linéaire | R² | 0.658 |

## A.2 Correspondance Finances ↔ Machine Learning

| Concept Finance | Équivalent ML |
|-----------------|---------------|
| Probabilité de Défaut | Classification binaire |
| WoE (Weight of Evidence) | Feature engineering |
| IV (Information Value) | Feature importance |
| Scorecard | Modèle interprétable |
| PSI | Drift detection |
| Backtesting | Validation temporelle |

## A.3 Références

1. **Accords de Bâle II** - Cadre réglementaire international
2. **Lending Club** - Source des données
3. **365 Data Science** - Cours de référence
4. **Scikit-learn** - Documentation officielle

---

*Guide rédigé avec acronymes développés et exemples concrets pour faciliter la compréhension.*

*Dernière mise à jour : Janvier 2026*
