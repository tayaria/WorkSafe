# Dataset WorkSafe (synthétique)

Ce dossier contient un **dataset simulé** qui sert à entraîner et tester un modèle de **détection d'anomalie** (par exemple un autoencoder) avant d'avoir de vraies données enregistrées sur la genouillère.

Les données sont générées par `generer_dataset.py`. Elles imitent le **log CSV du firmware** (`LOG_CSV = 1`, 50 Hz) : un modèle entraîné ici pourra ensuite être réutilisé sur les vrais enregistrements de l'ESP32.

## Contenu

| Fichier | Rôle |
|---|---|
| `generer_dataset.py` | Script qui génère les 3 fichiers CSV |
| `dataset_train.csv` | Entraînement : foulées **normales uniquement** |
| `dataset_val.csv` | Validation : choix du seuil d'alerte et réglages |
| `dataset_test.csv` | Test final, sur des athlètes jamais vus |

## Les 3 jeux de données

Chaque athlète virtuel n'apparaît que dans **un seul** des 3 fichiers, pour tester le modèle sur de nouvelles personnes.

| Jeu | Athlètes | Séances | Lignes | Anomalies |
|---|---|---|---|---|
| train | 16 | 16 × 3 min normales | ≈ 144 000 | 0 % |
| val | 4 | par athlète : 2 min normales + 4 min avec épisodes | ≈ 72 000 | ≈ 14 % |
| test | 6 | par athlète : 3 min normales + 8 min avec fatigue + 5 min avec épisodes | ≈ 288 000 | ≈ 31 % |

Le jeu d'entraînement ne contient que du normal : un autoencoder apprend la foulée normale, puis signale tout ce qui s'en écarte.

## Colonnes

### Colonnes identiques au log de l'ESP32

| Colonne | Signification | Unité |
|---|---|---|
| `temps_ms` | Temps depuis le début de la séance | ms |
| `ax_tibia` | Accélération du tibia selon son axe long (≈ +1 g debout) | g |
| `ay_tibia` | Accélération avant / arrière | g |
| `az_tibia` | Accélération latérale | g |
| `gy_tibia` | Vitesse de rotation du tibia | °/s |
| `gy_cuisse` | Vitesse de rotation de la cuisse | °/s |
| `angle_genou` | Angle du genou (flexion > 0, hyperextension < 0) | ° |
| `temperature` | Température du genou (mise à jour à 2 Hz) | °C |
| `niveau` | Niveau calculé par les règles du firmware : 0 = OK, 1 = ATTENTION, 2 = DANGER | — |

### Colonnes de « vérité terrain » (absentes du vrai log)

| Colonne | Signification |
|---|---|
| `athlete_id` | Numéro de l'athlète virtuel |
| `session_id` | Numéro de la séance |
| `stride_id` | Numéro de la foulée dans la séance (utile pour découper foulée par foulée) |
| `etiquette` | Type de situation : `normal`, `fatigue`, `instabilite`, `faux_pas`, `hyperextension`, `boiterie` |
| `anomalie` | 0 = normal, 1 = anomalie |
| `severite` | Intensité de l'anomalie, de 0 à 1 |

Une foulée est étiquetée anormale seulement si sa sévérité atteint **0,2**. En dessous, elle reste `normal` : c'est le début d'une fatigue encore à peine visible.

> **Attention** : ne pas donner ces colonnes au modèle en entrée. Elles servent uniquement à **évaluer** ses résultats.

## Types d'anomalies simulées

| Type | Ce qui change dans le signal |
|---|---|
| `fatigue` | S'installe progressivement : chocs plus forts, flexion plus faible, cadence plus lente, foulées plus irrégulières, échauffement |
| `instabilite` | Oscillations et chocs latéraux, vibrations du genou |
| `faux_pas` | 1 ou 2 foulées : choc brutal, rythme cassé |
| `hyperextension` | Le genou passe sous 0° en fin d'extension |
| `boiterie` | Amplitude réduite, cadence plus lente, choc accru |

## Régénérer les données

Prérequis : Python 3 avec `numpy` et `pandas`.

```bash
pip install numpy pandas
python generer_dataset.py . 42
```

- 1er argument : dossier de sortie (par défaut `.`)
- 2e argument : graine aléatoire (par défaut `42`). Une même graine redonne exactement les mêmes fichiers.

## Limites

- Les données sont **simulées** : elles servent à mettre au point le pipeline, pas à valider le modèle. Il faudra le réentraîner ou l'ajuster sur de vrais enregistrements.
- Les axes du MPU supposés (`ax` = axe long du tibia, `az` = latéral) doivent correspondre au montage réel du capteur.
- La colonne `niveau` est une approximation des règles du firmware, sans le délai de confirmation (0,5 s / 2 s). Elle sert à comparer le modèle aux règles, **pas** à l'entraîner.
- La rotation latérale du tibia n'est pas enregistrée dans le log. Elle compte dans `niveau`, mais le modèle ne la voit pas directement.
