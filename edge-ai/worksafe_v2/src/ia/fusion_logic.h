// fusion_logic.h — logique de fusion autoencoder + CNN (WorkSafe).
// Généré par fusion/fusion_v2.ipynb (fusion/firmware.py). Aucune allocation dynamique ; quelques dizaines d'octets d'état par séance.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "fusion_config.h"

struct FusionEtat {
  uint8_t n_ae;          // fenêtres consécutives au-dessus du seuil AE
  uint8_t n_cnn;         // fenêtres consécutives où une classe fiable dépasse son tau
  int8_t c_prec;         // classe de cette suite (-1 = aucune)
  uint8_t n_ok;          // fenêtres OK depuis la dernière alerte
  bool en_evenement;
  int8_t tampon[FUSION_V_STAB];  // types bruts des dernières fenêtres d'alerte de l'événement
  uint8_t n_tampon;
  uint8_t i_tampon;
#if FUSION_DERIVE
  uint16_t n_vu;
  float somme_ref;
  float recent[FUSION_N_RECENT];
  uint8_t n_rec;
  uint8_t i_rec;
#endif
};

struct FusionSortie {
  uint8_t etat;          // FUSION_ETAT_OK, FUSION_ETAT_ALERTE ou FUSION_ETAT_NON_CLASSEE
  int8_t type;           // 1..5 si FUSION_ETAT_ALERTE (ordre de CNN_CLASSES), 0 sinon
  int8_t classe_cnn;     // argmax du CNN sur la fenêtre (information)
  float proba_cnn;       // probabilité de cette classe
};

// À appeler au début de chaque séance (et après toute interruption du flux de fenêtres).
void fusion_reinitialiser(FusionEtat* e);
// Une fenêtre : score de l'autoencoder, dépassement brut (score >= AE_SEUIL_COMMUN), logits int8 du CNN.
void fusion_pas(FusionEtat* e, float score_ae, bool depasse_ae, const int8_t logits_q[6], FusionSortie* s);

#ifdef FUSION_VERIFIER_VECTEURS
// Rejoue fusion_test_vectors.h. Renvoie le nombre de pas en écart (0 = OK).
int fusion_verifier_vecteurs_test();
#endif
