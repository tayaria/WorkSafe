# WorkSafe — tester l'IA sur la carte (edge-ai)

Ce guide contient **toutes** les étapes. Vous n'avez pas besoin d'ouvrir un autre document.

## 1. C'est quoi ?

- L'IA de WorkSafe (un **autoencoder**, un **CNN** et une **logique de fusion**) a été ajoutée au firmware de la genouillère.
- Elle est en **mode ombre** : elle calcule et affiche ses résultats, mais elle ne commande **ni les LED, ni le moteur,
  ni le buzzer**. Le firmware marche comme avant.
- **Rien n'a encore tourné sur une carte** : tout a été compilé et vérifié sur PC seulement.
- Ce test sert à vérifier que **la chaîne technique marche** sur la carte, **pas** la qualité de la détection.
- **Aucune mesure de courant** n'est demandée.

## 2. Ce qu'il faut avoir

- un PC Windows ;
- la carte WorkSafe (avec ses deux capteurs) ;
- un câble USB **de données** (certains câbles ne font que charger : s'il n'apparaît aucun port, essayer un autre câble) ;
- le dépôt dans un **dossier court**, par exemple `C:\WorkSafe` (Windows limite la longueur des chemins ; un dossier
  trop profond fait échouer la compilation).

C'est tout. **Python n'est pas nécessaire.**

Durée : environ 1 heure, dont 15 minutes pour la première compilation.

---

## 3. Étape A : installation (une seule fois)

| élément | version EXACTE |
|---|---|
| IDE Arduino | 2.x (2.3 ou plus récente de la série 2) |
| paquet de cartes « esp32 » d'Espressif | **3.3.12** (ne jamais mettre à jour) |
| bibliothèque TensorFlow Lite Micro | **Chirale_TensorFLowLite 2.0.0** (attention : « TensorF**L**owLite », avec un L majuscule) |
| carte | **ESP32 Dev Module** (à confirmer, voir A5) |
| schéma de partition | **Huge APP (3MB No OTA/1MB SPIFFS)** : OBLIGATOIRE |
| moniteur série | **115200** bauds |

1. **Installer l'IDE Arduino 2.** Le télécharger sur https://www.arduino.cc/en/software (Windows), l'installer, l'ouvrir.
   *Vous devez voir :* une fenêtre avec un programme vide (`void setup()` ... `void loop()`).

2. **Ajouter l'adresse des cartes ESP32.** Menu **Fichier > Préférences**. Dans « URL de gestionnaire de cartes
   supplémentaires », coller : `https://espressif.github.io/arduino-esp32/package_esp32_index.json`, puis **OK**.

3. **Installer le paquet esp32 version 3.3.12.** Menu **Outils > Carte > Gestionnaire de carte**. Chercher « esp32 ».
   Dans « esp32 par Espressif Systems », choisir **3.3.12** dans la liste des versions, puis **Installer** (plusieurs minutes).
   *Vous devez voir :* « 3.3.12 installed ».
   **Ne cliquez jamais sur « Mettre à jour »** pour ce paquet, même si l'IDE le propose.

4. **Installer la bibliothèque.** Menu **Outils > Gérer les bibliothèques**. Chercher **Chirale**. Dans
   « Chirale_TensorFLowLite par Chirale, TensorFlow Authors », choisir **2.0.0**, puis **Installer** (accepter les
   dépendances si l'IDE en propose).
   *Vous devez voir :* « 2.0.0 installed ».

5. **Choisir la carte.** Menu **Outils > Carte > esp32 > ESP32 Dev Module**.
   La carte du projet est un ESP32 « classique » (module ESP32-WROOM-32 en général). Si votre carte porte un autre
   module (WROVER, S3, C3...), le signaler avant d'aller plus loin.

6. **Choisir la partition.** Menu **Outils > Partition Scheme > Huge APP (3MB No OTA/1MB SPIFFS)**. C'est obligatoire :
   avec le choix par défaut, le programme remplit 98 % de la place.

7. **Brancher la carte et choisir le port.** Brancher le câble USB. Menu **Outils > Port** : choisir le port COM qui
   apparaît (par exemple COM5).
   Si aucun port n'apparaît : installer le pilote de la petite puce USB de la carte, **CP210x**
   (https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers) ou **CH340/CH341**
   (https://www.wch-ic.com/downloads/CH341SER_EXE.html), puis débrancher et rebrancher.

8. **Ouvrir le bon sketch.** Menu **Fichier > Ouvrir** → **`edge-ai/worksafe_v2/worksafe_v2.ino`**, et UNIQUEMENT
   celui-ci (le `worksafe_v2.ino` à la racine du dépôt est l'ancienne version sans IA).
   Ne supprimez pas le fichier `build_opt.h` qui est à côté : il contient des réglages de compilation indispensables.

9. **Vérifier la compilation.** Cliquer sur **Vérifier** (le bouton ✓). La première fois, cela dure **5 à 15 minutes**
   (la bibliothèque TensorFlow Lite Micro est grosse).
   *Vous devez voir :* « Le croquis utilise 1294951 octets (41%) de l'espace de stockage de programmes » et « Les variables
   globales utilisent 111556 octets (34%) de mémoire dynamique » (les chiffres peuvent varier très légèrement).

## 4. Étape B : téléverser et lire l'auto-test

Ce test marche **même sans capteurs branchés**.

1. Cliquer sur **Téléverser** (la flèche →).
   *Vous devez voir :* « Writing at ... (100 %) » puis « Hard resetting via RTS pin... ».
2. Menu **Outils > Moniteur série**. En bas à droite : **115200 baud** et « Nouvelle ligne ».
3. Appuyer sur le bouton **EN** (ou RST) de la carte pour la redémarrer.
4. *Vous devez voir,* avant même les messages des capteurs :
   ```
   # IA prête (mode ombre) : arena autoencoder ... o, CNN ... o, tas libre ... o (avant : ... o)
   # IA auto-test détail : autoencoder 6020/6020, CNN 3792/3792, fusion 1854/1854 comparaisons justes (... ms)
   # IA auto-test : OK (11666 comparaisons, 0 écart)
   ```
   puis, si les capteurs sont branchés : `# WorkSafe pret : ...` et `# Calibration OK | ...`, et une ligne par seconde
   comme `K:3.2,V:12,P:1.1,S:0,C:0,T:31.5,D:0.0,N:0,F:000,A:-,Y:-,Z:-,I:-` (`A:-` : l'IA n'a encore rien calculé,
   c'est normal à l'arrêt).
5. **Si le message est différent** (ECHEC, « IA désactivée », « IA ERREUR », rien du tout) : copier **toute** la sortie
   du moniteur **telle quelle** (Ctrl+A puis Ctrl+C dans le moniteur), la mettre dans le rapport (étape F) et
   **s'arrêter là**. Ne rien corriger soi-même.

## 5. Étape C : les commandes

Taper une lettre dans la zone de saisie du moniteur série, puis **Entrée**.

| lettre | ce que ça fait | ce que vous devez voir | à noter dans le rapport |
|---|---|---|---|
| **T** | relance l'auto-test | `# IA auto-test : OK (11666 comparaisons, 0 écart)` | le résultat (OK ou ECHEC) |
| **B** | mesure la vitesse de calcul (200 calculs de chaque modèle) | 6 lignes `# B ...` : temps moyen, minimum et maximum en µs (millionièmes de seconde) de la préparation, de l'autoencoder, du CNN et de la fusion, puis la « charge estimée » en % | toutes les lignes `# B` |
| **I** | informations | 6 lignes `# I ...` : modèles, mémoire libre (« tas libre »), mémoire de travail des modèles (« arena »), marge de la pile de la tâche IA, fenêtres calculées et perdues, options | toutes les lignes `# I` |
| **P** | contrôle des capteurs sur la dernière minute | une ligne par capteur avec « OK » ou « SUSPECT », puis `# P résultat : ...` ; une ligne « aide » suit chaque capteur suspect | toutes les lignes `# P` |
| **L** | démarre (ou arrête) le journal de données | `# journal IA actif ...` puis beaucoup de lignes de chiffres (50 par seconde) ; à l'arrêt : `# journal IA arrêté` | rien : le journal s'enregistre dans un fichier (étape E) |

Dans la ligne de chaque seconde : `A:` = score de l'IA, `Y:` = type reconnu par le CNN, `Z:` = état de l'IA
(0 = OK, 1 = alerte avec un type, 2 = alerte sans type), `I:` = durée du calcul en millisecondes.

## 6. Étape D : le test avec les capteurs

1. Mettre la genouillère. Se tenir **debout, jambe tendue, immobile**, et appuyer sur **EN**. Ne pas bouger 3 secondes.
   *Vous devez voir :* `# Calibration OK | ...`.
2. Taper **B**, puis **I** : copier les lignes dans le rapport.
3. **Marcher 1 minute**, puis taper **P**. Copier les lignes `# P` dans le rapport (partie « marche »).
4. **Courir 1 minute** (sur place si besoin), puis taper **P**. Copier les lignes `# P` (partie « course »).
   Pendant la course, la ligne de chaque seconde doit montrer des nombres après `A:`, `Y:`, `Z:` et `I:`.
   Si un capteur est « SUSPECT », ne rien changer : copier simplement les lignes (le cas le plus probable est un
   angle du genou de signe inversé : ce sera corrigé ensuite).
5. **Enregistrer 5 minutes de course** avec la commande **L** : suivre l'étape E.

## 7. Étape E : enregistrer le journal dans un fichier

Le moniteur série de l'IDE ne sait pas enregistrer dans un fichier. Utiliser **PuTTY** (gratuit).

**Méthode PuTTY (recommandée)**
1. **Fermer le moniteur série de l'IDE** (un seul programme peut utiliser le port à la fois).
2. Installer PuTTY (https://www.putty.org) et l'ouvrir.
3. Page **Session** : cocher **Serial** ; « Serial line » : le port de la carte (ex. `COM5`) ; « Speed » : `115200`.
4. Dans la liste de gauche, **Session > Logging** : cocher **All session output** ; « Log file name » : **Browse**,
   choisir le Bureau et le nom `course_5min.txt`.
5. Revenir à **Session** et cliquer **Open**.
   *Vous devez voir :* une fenêtre noire avec une ligne par seconde.
6. Appuyer sur **EN** de la carte, rester **debout, jambe tendue, immobile** 3 secondes (calibration).
7. Cliquer dans la fenêtre PuTTY et taper **L** (une seule fois).
   *Vous devez voir :* `# journal IA actif ...` puis beaucoup de lignes de chiffres qui défilent.
8. **Courir 5 minutes.** Noter à la main l'heure de début, de fin, et les moments particuliers (arrêt, faux pas,
   fatigue ressentie...).
9. À la fin, taper **L** (*vous devez voir :* `# journal IA arrêté`), puis **I**.
10. Fermer PuTTY. Le fichier `course_5min.txt` est sur le Bureau.

**Méthode avec Python (optionnelle)** : si Python est installé, `pip install pyserial`, puis depuis le dossier du dépôt :
`python edge-ai/outils/capturer_serie.py COM5 --commande L` (Ctrl+C pour arrêter). Le fichier est écrit dans
`edge-ai/journaux/`.

## 8. Étape F : le rapport à renvoyer

1. Copier `edge-ai/worksafe_v2/RAPPORT_TEST_CARTE_modele.md` sous un nouveau nom, par exemple
   `RAPPORT_TEST_CARTE_2026-10-25.md`, et le remplir (ouvrable avec le Bloc-notes).
2. Renvoyer :
   - le rapport rempli (`RAPPORT_TEST_CARTE_<date>.md`) ;
   - le journal de la course (`course_5min.txt`, ou le fichier de `edge-ai/journaux/`) ;
   - en cas de problème : la sortie complète du moniteur série (copiée dans le rapport ou dans un fichier texte) et,
     si utile, une photo du câblage.

## 9. À quoi s'attendre

- L'IA a appris sur des données **simulées**. Sur de vrais signaux, l'autoencoder peut **alerter souvent** (`Z:1` ou
  `Z:2`) : c'est **attendu**, ce n'est **pas** un bug. Comme l'IA est en mode ombre, cela n'a aucun effet sur les LED,
  le moteur ou le buzzer.
- L'IA ne calcule que **pendant la course** (quand des foulées sont détectées) et seulement si le capteur de la cuisse
  est branché.

## 10. Dépannage

| ce que vous voyez | quoi faire |
|---|---|
| « Sketch too big » ou « text section exceeds available space » | Outils > Partition Scheme > **Huge APP** (étape A6) |
| « Chirale_TensorFlowLite.h: No such file or directory » | installer la bibliothèque (étape A4) |
| beaucoup d'erreurs dans les fichiers `esp32` ou `BLE` | mauvaise version du paquet : il faut **3.3.12** (étape A3) |
| « opening dependency file ... No such file or directory » | chemin trop long : mettre le dépôt dans un dossier court, par exemple `C:\WorkSafe` |
| aucun port COM dans Outils > Port | installer le pilote USB CP210x ou CH340 (étape A7), ou changer de câble |
| « # MPU tibia (0x68) introuvable » en boucle | câblage du capteur du tibia : SDA sur GPIO 21, SCL sur GPIO 22, AD0 relié à GND (l'auto-test, lui, s'affiche quand même avant) |
| « # IA désactivée (cuisse absente) » | le capteur de la cuisse (AD0 relié au 3V3) n'est pas vu : vérifier son câble ; le firmware marche sans l'IA |
| auto-test « ECHEC » | ne rien changer : renvoyer la sortie **telle quelle** (étape F) |
| la carte redémarre en boucle (« Guru Meditation », « rst:0x... », « Backtrace ») | copier les 20 lignes autour du message dans le rapport |
| rien ne s'affiche dans le moniteur | vérifier 115200 baud et le bon port COM ; appuyer sur EN |
| « Port occupé » / « Access is denied » | fermer le moniteur série de l'IDE ou PuTTY (un seul à la fois) |
| les lettres T, B, I, P, L ne font rien | taper la lettre puis **Entrée** dans la zone de saisie du moniteur |

---

## 11. Pour le développeur

| chemin | rôle |
|---|---|
| `RAPPORT_INTEGRATION.md` | environnement trouvé, bibliothèque choisie, tailles compilées, vérification `-ffp-contract=off`, tests, risques |
| `ADAPTATIONS.md` | ce qui diffère de `firmware_pack/` (aucun fichier copié n'est modifié) |
| `tests_pc/` | tests sur PC : rejeu C++ des calculs (`rejouer.py`), tests pytest (`test_edge_ai.py`) |
| `outils/` | `analyser_log.py` (rejoue un journal de la carte sur PC), `capturer_serie.py`, `journal_synthetique.py`, `synchroniser_pack.py` (copie depuis `firmware_pack/` et `MANIFEST.json`) |
| `MANIFEST.json` | SHA-256 de chaque fichier du package |
| `modeles/` | les deux modèles `.tflite` de référence (la carte utilise leurs tableaux C de `worksafe_v2/src/ia/`) |

Analyser un journal reçu (depuis la racine du dépôt, environnement `data/.venv`) :
`python edge-ai/outils/analyser_log.py <journal>`

## 12. Limites

Rien n'a été exécuté sur la carte. Ne sont **pas** vérifiés : TensorFlow Lite Micro à l'exécution, les temps de calcul,
la mémoire réellement utilisée, le comportement avec les vrais capteurs, la consommation électrique.
