// fusion_config.h — généré par fusion/fusion_v2.ipynb (fusion/firmware.py, depuis fusion/artifacts/config_fusion_v2.json).
// Ne pas modifier à la main. Règle de fusion autoencoder + CNN, choisie sur val + cnn_val.
// Données d'entraînement SIMULÉES : réglages à revalider sur de vraies données.
#pragma once
#include <stdint.h>

// Règle retenue : R2 m_ae=2 m_cnn=1 + dérive fatigue delta=0.3
// ---- Autoencoder : « il y a un problème » ----
#define FUSION_UTILISER_AE 1
#define FUSION_M_AE 2                 // alerte si M_AE fenêtres consécutives dépassent le seuil commun

// ---- CNN : « de quel type » ----
#define FUSION_NOMMER 1               // 1 = le CNN nomme le type des alertes
#define FUSION_TAU_NOM 0.0f          // type nommé seulement si probabilité CNN >= TAU_NOM, sinon « non classée »
// Classes fiables qui déclenchent SEULES (probabilité >= tau pendant M_CNN fenêtres) ; < 0 = jamais seule.
// Classes fiables : 2, 3, 4, 5
static const float FUSION_TAU_SEUL[6] = {-1.0f, -1.0f, 0.800000012f, 0.899999976f, 0.800000012f, 0.899999976f};
#define FUSION_M_CNN 1

// ---- Événements et stabilisation du type ----
#define FUSION_ECART_EVENEMENT 2     // un événement se termine après plus de 2 fenêtres OK
#define FUSION_V_STAB 1                // type affiché = classe la plus fréquente des V dernières fenêtres d'alerte

// ---- Dérive relative (fatigue lente) ----
// Alerte « fatigue » si (moyenne des N_RECENT dernières fenêtres) - (moyenne des N_REF premières fenêtres de la
// séance) >= FUSION_DERIVE_DELTA, pour le signal FUSION_DERIVE_SIGNAL.
#define FUSION_DERIVE 1                // 1 = option active
#define FUSION_SIGNAL_AE 1             // score de l'autoencoder : ANCIENNE dérive, écartée (réagit à toute anomalie, alertes longues)
#define FUSION_SIGNAL_FATIGUE 2        // probabilité « fatigue » du CNN (sortie de cnn_softmax, float)
#define FUSION_DERIVE_SIGNAL 2         // signal retenu : probabilité fatigue du CNN
#define FUSION_DERIVE_DELTA 0.300000012f
#define FUSION_DERIVE_AE_AUTORISEE 0    // 0 = l'ancienne dérive sur le score de l'AE est désactivée (refus à la compilation)
#define FUSION_N_REF 120              // référence : moyenne des N_REF premières fenêtres de la séance
#define FUSION_N_RECENT 60            // récent : moyenne des N_RECENT dernières fenêtres
#if FUSION_DERIVE && FUSION_DERIVE_SIGNAL == FUSION_SIGNAL_AE && !FUSION_DERIVE_AE_AUTORISEE
#error "dérive sur le score de l'autoencoder désactivée (voir fusion/rapport_fusion_v2.md)"
#endif

// ---- Sorties ----
#define FUSION_ETAT_OK 0
#define FUSION_ETAT_ALERTE 1            // type dans FusionSortie::type (1..5)
#define FUSION_ETAT_NON_CLASSEE 2
