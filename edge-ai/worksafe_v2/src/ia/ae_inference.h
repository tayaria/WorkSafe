// ae_inference.h — inférence de l'autoencoder WorkSafe avec TensorFlow Lite Micro.
// Généré par autoencoder/export_tflite.ipynb. Code de référence NON COMPILÉ sur carte : voir LISEZMOI.md.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "ae_config.h"

struct AeResultat {
  float score;    // score d'anomalie de la fenêtre
  float seuil;    // seuil appliqué (commun, ou calibré pour l'athlète)
  bool depasse;   // score >= seuil sur cette fenêtre
  bool alerte;    // alerte après la règle de persistance
  bool calibre;   // true si le seuil de l'athlète est disponible
};

// Charge le modèle et alloue l'arena. Renvoie false (et écrit un message) en cas d'erreur.
bool ae_init();
// À appeler au début de chaque séance : remet le compteur de persistance à zéro.
void ae_nouvelle_seance();
// Les AE_N_CALIB prochaines fenêtres (course normale) servent à calculer le seuil de l'athlète.
void ae_demarrer_calibration();
// Traite une fenêtre de AE_W x AE_N_CANAUX mesures BRUTES (unités du log CSV : g, deg/s, deg),
// dans l'ordre des canaux de ae_config.h. Renvoie false si l'inférence échoue.
bool ae_traiter_fenetre(const float brut[AE_W][AE_N_CANAUX], AeResultat* res);
// Octets d'arena réellement utilisés (après ae_init) : pour ajuster AE_ARENA_SIZE.
size_t ae_arena_utilisee();

// ---- Fonctions de calcul pur (sans TFLite Micro), utilisées aussi par les vecteurs de test ----
void ae_preparer_entree(const float brut[AE_W][AE_N_CANAUX], float z[AE_N_VALEURS], int8_t q[AE_N_VALEURS]);
float ae_calculer_score(const float z[AE_N_VALEURS], const int8_t sortie_q[AE_N_VALEURS]);
float ae_mediane(float* v, int n);  // trie v sur place

#ifdef AE_VERIFIER_VECTEURS
// Compare le firmware aux vecteurs de ae_test_vectors.h. Renvoie le nombre d'échecs (0 = OK).
int ae_verifier_vecteurs_test();
#endif
