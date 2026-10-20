# Rapport de test sur la carte — WorkSafe edge-ai (à remplir)

Copier ce fichier (par exemple `RAPPORT_TEST_CARTE_2026-10-25.md`), remplir chaque case, puis l'envoyer avec le
fichier journal de la course. Mettre les sorties du moniteur série **telles quelles**, entre les lignes ```.

## 1. Qui, quand, avec quoi

| champ | valeur |
|---|---|
| date | |
| personne | |
| carte (nom écrit sur la carte, module : ESP32-WROOM-32 ? autre ?) | |
| version de l'IDE Arduino (menu Aide > À propos) | |
| version du paquet esp32 (doit être 3.3.12) | |
| version de la bibliothèque Chirale_TensorFLowLite (doit être 2.0.0) | |
| schéma de partition | [ ] Huge APP  [ ] autre : |
| taille du programme : « Le croquis utilise ... octets (...%) » | |
| RAM : « Les variables globales utilisent ... octets (...%) » | |

## 2. Auto-test au démarrage (étape 2)

Résultat : [ ] OK   [ ] ECHEC   [ ] pas de message `# IA ...`

```
(coller ici toutes les lignes commençant par # après le redémarrage)
```

## 3. Banc de mesure : commande B (étape 3)

```
(coller les lignes # B ...)
```

## 4. Informations : commande I (étape 4)

```
(coller les lignes # I ...)
```

## 5. Plausibilité des capteurs : commande P (étape 5)

Marche (1 minute) :
```
(coller les lignes # P ...)
```

Course (1 minute) :
```
(coller les lignes # P ...)
```

## 6. Course de 5 minutes en mode ombre (étape 6)

| question | réponse |
|---|---|
| nom du fichier journal joint | |
| heure de début / de fin | |
| la carte a-t-elle redémarré pendant la course ? (messages « rst: », « Guru Meditation ») | [ ] non  [ ] oui : à quel moment ? |
| les LED / le moteur / le buzzer ont-ils marché comme d'habitude ? | [ ] oui  [ ] non : |
| l'IA a-t-elle alerté (`Z:1` ou `Z:2` dans la ligne de chaque seconde) ? | [ ] jamais  [ ] parfois  [ ] presque tout le temps |
| à quels moments (approximatifs) ? | |
| moments particuliers notés (arrêt, faux pas, fatigue ressentie...) | |
| lignes `# I ...` après la course (fenêtres perdues) | |

## 7. Remarques libres

(tout ce qui a semblé bizarre, même petit)

## 8. Fichiers joints

- [ ] ce rapport rempli
- [ ] le fichier journal de la course (PuTTY ou `capturer_serie.py`)
- [ ] (si problème) une photo du câblage
