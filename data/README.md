# Dataset WorkSafe (synthétique)

Ce dossier contient un **dataset simulé** qui sert à mettre au point deux modèles qui travaillent ensemble, avant d'avoir de vraies données enregistrées sur la genouillère :

- un **autoencoder** qui dit « il y a un problème » (détection d'anomalie, entraîné sur du normal seul) ;
- un **CNN 1D** qui dit « de quel type » : `normal`, `fatigue`, `instabilite`, `faux_pas`, `hyperextension`, `boiterie`.

Les deux modèles devront tourner sur l'ESP32 (TFLite int8).

Les données sont générées par `generer_dataset.py`. Elles imitent le **log CSV du firmware** (`LOG_CSV = 1`, 50 Hz) : un modèle entraîné ici pourra ensuite être réutilisé sur les vrais enregistrements de l'ESP32.

## Contenu

| Fichier | Rôle |
|---|---|
| `generer_dataset.py` | Script qui génère les 7 fichiers CSV |
| `dataset_train.csv` | Autoencoder, entraînement : foulées **normales uniquement** |
| `dataset_val.csv` | Autoencoder, validation : choix du seuil d'alerte et réglages |
| `dataset_test.csv` | Autoencoder, test final sur des athlètes jamais vus |
| `dataset_cnn_train.csv` | CNN, entraînement : toutes les classes, nouveaux athlètes |
| `dataset_cnn_val.csv` | CNN, validation : réglages et arrêt de l'entraînement |
| `dataset_cnn_test.csv` | CNN, test final ; aussi test final de la combinaison autoencoder + CNN |
| `dataset_cnn_test2.csv` | **Confirmation finale du prototype** (autoencoder + CNN + fusion) sur 10 athlètes nouveaux ; **jamais utilisé pour décider** |
| `explore_data.py` | Exploration des 6 fichiers → `exploration/rapport_exploration.md` |
| `build_windows.py` | Fenêtres de 1 s + normalisation → `windows/` (voir `windows/rapport_fenetres.md`) ; `--ajouter cnn_test2` : fenêtres du nouveau jeu avec la normalisation **existante**, sans réécrire aucun autre fichier |
| `tests/` | Tests rapides de `build_windows.py` : `python -m pytest tests -q` |
| `requirements.txt` | Paquets Python nécessaires |

## Quel fichier pour quel modèle

| Modèle | Entraînement | Validation | Test |
|---|---|---|---|
| Autoencoder (« il y a un problème ») | `dataset_train.csv` | `dataset_val.csv` | `dataset_test.csv` |
| CNN 1D (« de quel type ») | `dataset_cnn_train.csv` | `dataset_cnn_val.csv` | `dataset_cnn_test.csv` |
| Combinaison (autoencoder puis CNN) | — | — | `dataset_cnn_test.csv`, puis confirmation finale sur `dataset_cnn_test2.csv` |

Le **test final de la combinaison** se fait sur `dataset_cnn_test.csv` : ses 8 athlètes n'ont été vus par aucun des deux modèles, et il contient assez d'exemples de chaque classe (au moins 150 fenêtres de 1 s par type d'anomalie) pour que le score par classe ait un sens.

## Les 7 jeux de données

Chaque athlète virtuel n'apparaît que dans **un seul** des 7 fichiers, pour tester les modèles sur de nouvelles personnes.

| Jeu | Athlètes (ids) | Séances par athlète | Lignes | Anomalies |
|---|---|---|---|---|
| train | 16 (1-16) | 3 min normales | ≈ 144 000 | 0 % |
| val | 4 (17-20) | 2 min normales + 4 min avec épisodes | ≈ 72 000 | ≈ 14 % |
| test | 6 (21-26) | 3 min normales + 8 min avec fatigue + 5 min avec épisodes | ≈ 288 000 | ≈ 31 % |
| cnn_train | 20 (27-46) | 3 min normales + 8 min avec épisodes + fatigue (5, 6, 7 ou 8 min) + 3 min « épisodes rares » | ≈ 1 231 000 | ≈ 31 % |
| cnn_val | 5 (47-51) | idem cnn_train | ≈ 303 000 | ≈ 31 % |
| cnn_test | 8 (52-59) | 3 min normales + 8 min avec épisodes + fatigue (5 ou 8 min) + 3 min « épisodes rares » | ≈ 493 000 | ≈ 31 % |
| cnn_test2 | 10 (60-69) | 2 × 4 min normales + 8 min avec épisodes + 3 min « épisodes rares » + fatigue (5 ou 8 min, en alternance) | ≈ 766 000 | ≈ 25 % |

Le jeu d'entraînement de l'autoencoder ne contient que du normal : l'autoencoder apprend la foulée normale, puis signale tout ce qui s'en écarte.

### Particularités des jeux CNN

Chaque athlète des fichiers `cnn_*` a 4 séances, dans cet ordre (colonne `session_type`) :

| `session_type` | Durée | Contenu |
|---|---|---|
| `normal` | 3 min | Foulées normales uniquement |
| `episodes` | 8 min | Épisodes courts des 4 types, **à tour de rôle** selon le cycle `instabilite, faux_pas, hyperextension, faux_pas, boiterie, faux_pas` (un faux pas ne dure que 1 à 2 foulées, il revient donc 3 fois par cycle). Écart entre épisodes : **12-30 foulées** (au lieu de 25-55 dans val / test) |
| `fatigue` | variable | Fatigue progressive. cnn_train et cnn_val : 5, 6, 7 ou 8 min à tour de rôle d'un athlète à l'autre ; cnn_test : la moitié des athlètes à 5 min, l'autre moitié à 8 min |
| `episodes_rares` | 3 min | Surtout des faux pas (3 épisodes sur 4, cycle `faux_pas, faux_pas, hyperextension, faux_pas`), le reste en hyperextension. Écart entre épisodes plus court : **6-12 foulées** |

Dans tous les fichiers, un faux pas dure 1 ou 2 foulées et **deux faux pas sont toujours séparés par au moins 6 foulées** : un faux pas reste isolé, c'est ce qui le distingue d'une boiterie ou d'une fatigue. Le script vérifie cette règle et s'arrête avec une erreur si elle n'est pas respectée. La physique du simulateur (les signatures ci-dessous) est la même pour tous les fichiers.

Fenêtres de 1 s (50 points, pas de 25 points, règle majoritaire) par type d'anomalie :

| Fichier | instabilite | faux_pas | hyperextension | boiterie | Cible minimale |
|---|---|---|---|---|---|
| cnn_train | 1 723 | 898 | 3 230 | 1 634 | 600 |
| cnn_val | 392 | 208 | 815 | 442 | 150 |
| cnn_test | 637 | 377 | 1 313 | 629 | 150 |

`normal` reste majoritaire (≈ 69 % des lignes) et `faux_pas` reste la classe la plus rare (≈ 1,8 % des lignes).

Ces fichiers sont produits avec un générateur aléatoire séparé (graine + 1), appelé après les 3 premiers fichiers : `dataset_train`, `dataset_val` et `dataset_test` restent identiques pour une même graine.

### Jeu de confirmation finale : `dataset_cnn_test2.csv`

**Rôle** : confirmer le prototype complet (autoencoder + CNN + logique de fusion) sur des athlètes **nouveaux**, après
que cnn_test et test ont été trop regardés pour rester des mesures vierges. **Il ne sert jamais à décider** (ni seuil,
ni règle, ni paramètre) : on y mesure une seule fois la règle déjà gelée, et si un critère n'est pas atteint, on le dit
sans re-régler sur ce jeu.

- 10 athlètes **nouveaux** (ids 60-69), absents de tous les autres fichiers ; mêmes colonnes que `dataset_cnn_test.csv`
  (dont `session_type`) ; même physique et même variabilité entre athlètes que les autres jeux.
- Par athlète, 5 séances : **2 séances normales de 4 min** (pour mesurer les fausses alertes sur plus de temps :
  80 min au total), 1 séance `episodes` de 8 min (types à tour de rôle), 1 séance `episodes_rares` de 3 min (surtout
  des faux pas, au moins 6 foulées entre deux faux pas), 1 séance `fatigue` (5 et 8 min en alternance d'un athlète à l'autre).
- Générateur aléatoire **séparé** (graine + 2), appelé **après** tous les autres fichiers : les 6 fichiers existants
  restent identiques octet pour octet pour une même graine (vérifié par leur SHA-256).
- Fenêtres : `windows/cnn_test2.npz` et `windows/cnn_test2_meta.csv`, construites avec `python build_windows.py . --ajouter cnn_test2`
  et la normalisation **existante** (`windows/normalisation.json`, ni recalculée ni modifiée), avec les mêmes contrôles
  automatiques (pas de traversée de séance, pas de NaN, athlètes disjoints, `y_maj` / `y_bin` / `ambigu` recalculés
  indépendamment). 30 562 fenêtres, dont 6,5 % ambiguës.

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
| `session_type` | Type de séance : `normal`, `episodes`, `fatigue` ou `episodes_rares` (**fichiers `cnn_*` seulement**, pour l'analyse ; ne pas donner au modèle) |

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

Prérequis : Python 3 avec `numpy` et `pandas` (voir `requirements.txt`).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux / macOS : source .venv/bin/activate)
pip install -r requirements.txt
python generer_dataset.py . 42
```

À la fin, le script affiche un récapitulatif par fichier : nombre de lignes, d'athlètes et de lignes par étiquette.

- 1er argument : dossier de sortie (par défaut `.`)
- 2e argument : graine aléatoire (par défaut `42`). Une même graine redonne exactement les mêmes fichiers.

## Limites

- Les données sont **simulées** : elles servent à mettre au point le pipeline, pas à valider les modèles. Les signatures des anomalies sont écrites à la main et sont sans doute plus nettes et plus régulières que dans la réalité. Les bons scores obtenus ici ne prouvent rien sur de vraies données : il faudra réentraîner ou ajuster les modèles sur de vrais enregistrements.
- Les classes restent déséquilibrées, surtout pour le CNN (`faux_pas` ≈ 1,8 % des lignes, `normal` ≈ 69 %).
- La séance `episodes_rares` concentre beaucoup de faux pas en 3 min, ce qui n'arrive pas en vrai : elle sert à avoir assez d'exemples, pas à imiter une vraie séance. Le modèle ne doit pas recevoir `session_type`.
- Les athlètes virtuels sont tirés d'une même loi : 33 athlètes CNN ne représentent pas la diversité de vraies personnes (morphologie, chaussures, terrain, pose du capteur).
- Les axes du MPU supposés (`ax` = axe long du tibia, `az` = latéral) doivent correspondre au montage réel du capteur.
- La colonne `niveau` est une approximation des règles du firmware, sans le délai de confirmation (0,5 s / 2 s). Elle sert à comparer le modèle aux règles, **pas** à l'entraîner.
- La rotation latérale du tibia n'est pas enregistrée dans le log. Elle compte dans `niveau`, mais le modèle ne la voit pas directement.
