# WorkSafe v2

Firmware ESP32 pour une genouillère connectée qui surveille le genou d'un athlète pendant l'effort. Il détecte les situations à risque (flexion ou hyperextension excessive, chocs, instabilité latérale, échauffement, fatigue) et alerte par LED, vibration et buzzer. Les mesures sont envoyées en Bluetooth Low Energy.

Tout le code est dans un seul fichier, [`worksafe_v2.ino`](worksafe_v2.ino). Il n'utilise aucune bibliothèque externe.

## Fonctionnalités

- **Angle réel du genou** : différence d'inclinaison entre la cuisse et le tibia, calculée avec un filtre complémentaire (gyroscope + accéléromètre). Le firmware calcule aussi la vitesse angulaire et détecte l'hyperextension.
- **Foulées, chocs et cadence** : détectés par l'accéléromètre du tibia.
- **Fatigue** : comparée à une référence prise sur les 100 premières foulées. Trois signes sont suivis : les chocs augmentent, l'amplitude de flexion baisse, la cadence baisse.
- **Échauffement** : hausse de température (LM35) par rapport à la valeur mesurée à la calibration.
- **Instabilité latérale** : rotation du tibia hors du plan sagittal.
- **Décision** : niveau DANGER si un indicateur est critique **ou** si deux indicateurs sont en attention. Une montée de niveau doit durer 0,5 s pour être prise en compte, une descente 2 s.
- **Alertes** : LED RGB, moteur vibrant et buzzer. En DANGER, l'alerte est répétée toutes les 10 s.
- **BLE (Nordic UART Service)** : une trame par seconde, plus des commandes à distance.
- **Log CSV** sur le port série, pour constituer un dataset d'IA.

Si le MPU de la cuisse (0x69) est absent, le firmware passe en **mode 1 capteur** : l'angle est alors l'inclinaison du tibia.

## Matériel

| Composant | Rôle |
|---|---|
| ESP32 | Microcontrôleur, BLE |
| MPU-6050 n°1 (adresse 0x68, AD0 → GND) | Tibia (obligatoire) |
| MPU-6050 n°2 (adresse 0x69, AD0 → 3V3) | Cuisse (optionnel) |
| LM35 | Température |
| Moteur vibrant + transistor (résistance de base 1 kΩ) | Alerte haptique |
| Buzzer actif | Alerte sonore |
| 3 LED RGB | Niveau, état BLE, système |
| Batterie Li-Po + TP4056 | Alimentation et charge |
| Diviseur 100k/100k (optionnel) | Mesure de la tension batterie |

## Câblage

| Signal | GPIO ESP32 |
|---|---|
| I²C SDA (les 2 MPU) | 21 |
| I²C SCL (les 2 MPU) | 22 |
| LM35 (sortie) | 34 |
| Tension batterie (diviseur) | 35 |
| Moteur vibrant (via transistor) | 25 |
| Buzzer | 26 |
| LED1 R / G / B | 13 / 14 / 27 |
| LED2 R / G / B | 16 / 17 / 18 |
| LED3 R / G / B | 19 / 23 / 4 |

Les LED sont à cathode commune par défaut, avec une résistance sur la patte commune. Pour des LED à anode commune, mettre `LED_COMMON_ANODE` à 1.

## Installation

1. Installer l'[IDE Arduino](https://www.arduino.cc/en/software) et le paquet de cartes **esp32** d'Espressif (Gestionnaire de cartes).
2. Ouvrir `worksafe_v2.ino`.
3. Choisir la carte (par exemple *ESP32 Dev Module*) et le port.
4. Téléverser, puis ouvrir le moniteur série à **115200 bauds**.

## Utilisation

1. Au démarrage, l'athlète se tient **debout, jambe tendue et immobile pendant environ 2 s**. Une calibration automatique a lieu : biais des gyroscopes, angle de référence et température de repos. Un bip la signale au début, une vibration courte à la fin.
2. Si le MPU du tibia n'est pas détecté, la LED3 clignote en rouge, le buzzer bipe et le firmware réessaie en boucle.
3. Si l'angle **diminue** quand on fléchit le genou, mettre `KNEE_SIGN` à `-1`.

### Signification des LED

| LED | État |
|---|---|
| LED1 | Vert = OK, orange = ATTENTION, rouge clignotant = DANGER |
| LED2 | Bleu fixe = BLE connecté, bleu clignotant = en attente de connexion |
| LED3 | Rouge clignotant = erreur capteur, flash cyan = foulée détectée, blanc = calibration en cours |

## Bluetooth (BLE)

L'appareil s'annonce sous le nom **`WorkSafe`** avec le service Nordic UART. Il est compatible avec des applications comme *nRF Connect* ou *Serial Bluetooth Terminal*.

| Caractéristique | UUID |
|---|---|
| Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| RX (écriture, commandes) | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` |
| TX (notifications, trames) | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` |

### Commandes

| Commande | Action |
|---|---|
| `C` | Relancer la calibration |
| `M` | Couper ou rétablir le buzzer (mute) |

### Trame (1 Hz, BLE + série)

```
K:35.2,V:120,P:4.1,S:152,C:84,T:33.4,D:1.2,N:0,F:000
```

| Champ | Signification | Unité |
|---|---|---|
| `K` | Angle du genou (flexion > 0, hyperextension < 0) | ° |
| `V` | Vitesse angulaire du genou | °/s |
| `P` | Pic d'accélération de la dernière foulée | g |
| `S` | Nombre de foulées | — |
| `C` | Cadence de la jambe équipée | foulées/min |
| `T` | Température | °C |
| `D` | Écart de température par rapport au repos | °C |
| `N` | Niveau : 0 = OK, 1 = ATTENTION, 2 = DANGER | — |
| `F` | Indicateurs actifs (masque de bits, en hexadécimal) | — |
| `B` | Tension batterie (si `USE_BATT_SENSE` vaut 1) | V |

Bits du champ `F` :

| Bit | Indicateur |
|---|---|
| 0 (`0x001`) | Flexion excessive |
| 1 (`0x002`) | Hyperextension |
| 2 (`0x004`) | Vitesse angulaire du genou |
| 3 (`0x008`) | Choc à l'impact |
| 4 (`0x010`) | Échauffement |
| 5 (`0x020`) | Instabilité latérale |
| 6 (`0x040`) | Fatigue : hausse des chocs |
| 7 (`0x080`) | Fatigue : baisse d'amplitude de flexion |
| 8 (`0x100`) | Fatigue : baisse de cadence |

## Configuration

Les options se trouvent en tête du fichier :

| Option | Défaut | Description |
|---|---|---|
| `USE_MOTOR` | 1 | Moteur vibrant branché |
| `USE_RGB` | 1 | LED RGB branchées |
| `LED_COMMON_ANODE` | 0 | 0 = cathode commune, 1 = anode commune |
| `BUZZER_ON_WARN` | 0 | Le buzzer sonne aussi au niveau ATTENTION |
| `USE_BATT_SENSE` | 0 | Mesure de la batterie sur GPIO35 |
| `KNEE_SIGN` | 1 | Inverse le signe de l'angle du genou |
| `LOG_CSV` | 0 | Log CSV sur le port série (remplace la trame série) |
| `LOG_CSV_DIV` | 4 | 1 ligne CSV toutes les N mesures (4 → 50 Hz) |

### Seuils

Les seuils (section `SEUILS`) sont **à calibrer pour chaque athlète** :

| Indicateur | Attention | Danger |
|---|---|---|
| Flexion | 120° | 140° |
| Hyperextension | −5° | −10° |
| Vitesse angulaire du genou | 700 °/s | 900 °/s |
| Pic d'impact | 5 g | 8 g |
| Rotation latérale | 300 °/s | 500 °/s |
| Échauffement | +2,5 °C | +4 °C |
| Fatigue : hausse des chocs | +25 % | +50 % |
| Fatigue : baisse d'amplitude | −15 % | −25 % |
| Fatigue : baisse de cadence | −10 % | −20 % |

Une foulée est détectée au-dessus de 2 g (`STEP_G`) et un choc est compté au-dessus de 3 g (`IMPACT_G`). La référence de fatigue est fixée après `BASE_STRIDES` (100) foulées.

## Log CSV (dataset IA)

Avec `LOG_CSV` à 1, le port série émet à 50 Hz des lignes au format suivant :

```
temps_ms,ax_tibia,ay_tibia,az_tibia,gy_tibia,gy_cuisse,angle_genou,temperature,niveau
```

Les accélérations sont en g, les vitesses angulaires en °/s, l'angle en ° et la température en °C. Les lignes commençant par `#` sont des messages d'état et sont à ignorer lors du traitement.

## Détails techniques

- Boucle principale à **200 Hz**, CPU à 80 MHz pour économiser la batterie.
- MPU-6050 configurés à ±16 g et ±2000 °/s, avec un filtre passe-bas d'environ 184 Hz.
- I²C à 400 kHz.
- Température et batterie lues à 2 Hz.
