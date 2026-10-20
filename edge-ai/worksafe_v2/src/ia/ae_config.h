// ae_config.h — généré par autoencoder/export_tflite.ipynb. Ne pas modifier à la main.
// Autoencoder WorkSafe : constantes de prétraitement, de quantification, de score et de seuil.
// Données d'entraînement SIMULÉES : seuils et constantes à revalider sur de vraies données.
#pragma once
#include <stdint.h>

// ---- Fenêtre ----
#define AE_W 50                         // pas de temps par fenêtre (1 s à 50 Hz)
#define AE_N_CANAUX 6
#define AE_N_VALEURS (AE_W * AE_N_CANAUX)    // ordre en mémoire : [t][canal]
#define AE_PAS_FENETRE 25                 // nouvelle fenêtre tous les 25 échantillons (0,5 s)
// Ordre des canaux : "ax_tibia", "ay_tibia", "az_tibia", "gy_tibia", "gy_cuisse", "angle_genou"
static const char* const AE_CANAUX[AE_N_CANAUX] = {"ax_tibia", "ay_tibia", "az_tibia", "gy_tibia", "gy_cuisse", "angle_genou"};

// ---- Normalisation (data/windows/normalisation.json) : z = (x - MEAN) * INV_STD ----
static const float AE_MEAN[AE_N_CANAUX] = {0.835913897f, -0.542553306f, -1.29148093e-05f, 0.470247418f, 0.0209153369f, 36.9790535f};
static const float AE_STD[AE_N_CANAUX] = {0.773572028f, 0.444663525f, 0.0833529606f, 347.238586f, 161.805557f, 23.9611015f};
static const float AE_INV_STD[AE_N_CANAUX] = {1.29270446f, 2.24889135f, 11.9971743f, 0.0028798643f, 0.00618025707f, 0.0417343117f};  // 1 / STD (float32)
#define AE_CLIP 8.0f                   // écrêtage de z à +/- AE_CLIP

// ---- Quantification int8 (lue dans le .tflite) ----
#define AE_IN_SCALE 0.0627451017f
#define AE_IN_INV_SCALE 15.937499f  // 1 / AE_IN_SCALE (float32)
#define AE_IN_ZERO_POINT (-1)
#define AE_OUT_SCALE 0.0593841784f
#define AE_OUT_ZERO_POINT (-14)

// ---- Score d'anomalie (calculé hors du modèle, en float) ----
#define AE_SCORE_S1 1       // MSE moyen sur 50 x 6
#define AE_SCORE_S2 2       // moyenne des AE_S2_TOP_K pires pas de temps
#define AE_SCORE_S3_MOY 3   // MSE par canal / médiane du canal, moyenne sur les canaux
#define AE_SCORE_S3_MAX 4   // MSE par canal / médiane du canal, maximum sur les canaux
#define AE_SCORE_TYPE 3      // score retenu : S3_moy
#define AE_S2_TOP_K 5
static const float AE_S3_MEDIANES[AE_N_CANAUX] = {0.0482282043f, 0.0351990387f, 0.317049354f, 0.0133848526f, 0.00725839613f, 0.0136425523f};  // erreur médiane par canal (train_fit, modèle int8)

// ---- Seuils ----
#define AE_SEUIL_COMMUN 3.98210382f    // 99e centile des scores normaux de val (modèle int8)
#define AE_UTILISER_CALIBRATION 0            // 0 = seuil commun seul (décision d'équipe, voir LISEZMOI.md) ; 1 = seuil par athlète après calibration
#define AE_K_CALIB 2.01999998f                  // seuil athlète = AE_K_CALIB x médiane des scores de calibration
#define AE_N_CALIB 120                   // fenêtres de calibration (~1 min de course normale)
#define AE_PERSISTANCE 2                 // alerte si AE_PERSISTANCE fenêtres consécutives dépassent (1 = sans persistance)

// ---- Mémoire TFLite Micro (ESTIMATION, à ajuster avec ae_arena_utilisee() sur la carte) ----
#define AE_ARENA_SIZE (8 * 1024)
