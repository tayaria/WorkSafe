// cnn_config.h — généré par cnn/export_tflite.ipynb. Ne pas modifier à la main.
// CNN 1D WorkSafe (« de quel type ? ») : entrée, classes, quantification, vote, seuil de confiance.
// Données d'entraînement SIMULÉES : réglages à revalider sur de vraies données.
#pragma once
#include <stdint.h>

// ---- Fenêtre d'entrée : DÉJÀ NORMALISÉE (z = (x - moyenne) / écart-type, voir ae_config.h) ----
#define CNN_W 50                        // pas de temps (1 s à 50 Hz)
#define CNN_N_CANAUX 6
#define CNN_N_VALEURS (CNN_W * CNN_N_CANAUX)  // ordre en mémoire : [t][canal]
// Ordre des canaux : "ax_tibia", "ay_tibia", "az_tibia", "gy_tibia", "gy_cuisse", "angle_genou"
static const char* const CNN_CANAUX[CNN_N_CANAUX] = {"ax_tibia", "ay_tibia", "az_tibia", "gy_tibia", "gy_cuisse", "angle_genou"};
#define CNN_CLIP 8.0f                 // écrêtage de z à +/- CNN_CLIP avant quantification

// ---- Classes (ordre des sorties du modèle) ----
#define CNN_N_CLASSES 6
static const char* const CNN_CLASSES[CNN_N_CLASSES] = {"normal", "fatigue", "instabilite", "faux_pas", "hyperextension", "boiterie"};
#define CNN_INCERTAIN (-1)            // réponse « incertain » (option seuil de confiance)

// ---- Quantification int8 : réel = (q - ZERO_POINT) * SCALE ----
#define CNN_IN_SCALE 0.0627451017f
#define CNN_IN_INV_SCALE 15.937499f       // 1 / CNN_IN_SCALE (float32)
#define CNN_IN_ZERO_POINT (-1)
#define CNN_OUT_SCALE 0.198118418f       // sortie = logits
#define CNN_OUT_ZERO_POINT (-27)

// ---- Vote temporel causal (k dernières fenêtres de la séance) ----
#define CNN_VOTE_MAJORITE 1
#define CNN_VOTE_MOYENNE 2
#define CNN_K_VOTE 1                    // retard ajouté au plus (k - 1) x 0,5 s
#define CNN_REGLE_VOTE 1               // majorite

// ---- Seuil de confiance (option « incertain ») ----
#define CNN_TAU 0.99000001f               // confiance = probabilité moyenne de la classe retenue sur le vote
#define CNN_UTILISER_INCERTAIN 0        // 1 = répondre CNN_INCERTAIN si confiance < CNN_TAU

// ---- Mémoire TFLite Micro (ESTIMATION, à réduire avec cnn_arena_utilisee() sur la carte) ----
#define CNN_ARENA_SIZE (16 * 1024)
