// fusion_logic.cpp — logique de fusion autoencoder + CNN (WorkSafe).
// Généré par fusion/fusion_v2.ipynb (fusion/firmware.py). Référence Python exacte : fusion/firmware.py (MachineFusion).
// Calculs en float, dans le même ordre que la référence. Le softmax est celui du CNN (cnn_softmax),
// disponible sans TFLite Micro en compilant cnn_inference.cpp avec -DCNN_SANS_TFLM si besoin.
#include "fusion_logic.h"

#include <string.h>

#include "cnn_inference.h"

void fusion_reinitialiser(FusionEtat* e) { memset(e, 0, sizeof(*e)); e->c_prec = -1; }

void fusion_pas(FusionEtat* e, float score_ae, bool depasse_ae, const int8_t logits_q[6], FusionSortie* s) {
  float p[CNN_N_CLASSES];
  cnn_softmax(logits_q, p);
  int c = 0;
  for (int k = 1; k < CNN_N_CLASSES; ++k)
    if (p[k] > p[c]) c = k;
  s->classe_cnn = (int8_t)c;
  s->proba_cnn = p[c];

  // 1. autoencoder avec persistance
  e->n_ae = depasse_ae ? (e->n_ae < 255 ? e->n_ae + 1 : 255) : 0;
  const bool ae_alerte = FUSION_UTILISER_AE && e->n_ae >= FUSION_M_AE;

  // 2. CNN seul, pour les classes fiables
  const float tau = FUSION_TAU_SEUL[c];
  if (c != 0 && tau >= 0.0f && p[c] >= tau) {
    e->n_cnn = (c == e->c_prec) ? (e->n_cnn < 255 ? e->n_cnn + 1 : 255) : 1;
    e->c_prec = (int8_t)c;
  } else {
    e->n_cnn = 0;
    e->c_prec = -1;
  }
  const bool cnn_alerte = e->n_cnn >= FUSION_M_CNN;

  // 3. dérive relative (fatigue lente)
  bool derive_alerte = false;
#if FUSION_DERIVE
  const float sig = (FUSION_DERIVE_SIGNAL == FUSION_SIGNAL_AE) ? score_ae : p[1];
  if (e->n_vu < 65535) e->n_vu++;
  if (e->n_vu <= FUSION_N_REF) e->somme_ref += sig;
  e->recent[e->i_rec] = sig;
  e->i_rec = (uint8_t)((e->i_rec + 1) % FUSION_N_RECENT);
  if (e->n_rec < FUSION_N_RECENT) e->n_rec++;
  if (e->n_vu > FUSION_N_REF) {
    const int debut = ((int)e->i_rec - (int)e->n_rec + FUSION_N_RECENT) % FUSION_N_RECENT;
    float somme = 0.0f;
    for (int i = 0; i < e->n_rec; ++i) somme += e->recent[(debut + i) % FUSION_N_RECENT];  // plus ancien -> plus récent
    const float moy_rec = somme / (float)e->n_rec;
    const float moy_ref = e->somme_ref / (float)FUSION_N_REF;
    derive_alerte = (moy_rec - moy_ref) >= FUSION_DERIVE_DELTA;
  }
#else
  (void)score_ae;
#endif

  // 4. type brut (6 = alerte non classée)
  const bool nomme = FUSION_NOMMER && c != 0 && p[c] >= FUSION_TAU_NOM;
  int t;
  if (cnn_alerte) t = c;
  else if (ae_alerte && nomme) t = c;
  else if (derive_alerte) t = 1;
  else if (ae_alerte) t = 6;
  else t = 0;

  // 5. événement et stabilisation du type
  if (t == 0) {
    if (e->n_ok < 255) e->n_ok++;
    if (e->n_ok > FUSION_ECART_EVENEMENT) e->en_evenement = false;
    s->etat = FUSION_ETAT_OK;
    s->type = 0;
    return;
  }
  if (!e->en_evenement) {
    e->n_tampon = 0;
    e->i_tampon = 0;
  }
  e->en_evenement = true;
  e->n_ok = 0;
  e->tampon[e->i_tampon] = (int8_t)t;
  e->i_tampon = (uint8_t)((e->i_tampon + 1) % FUSION_V_STAB);
  if (e->n_tampon < FUSION_V_STAB) e->n_tampon++;
  int compte[7] = {0};
  int best = 0;
  for (int i = 0; i < e->n_tampon; ++i) {  // du plus récent au plus ancien : à égalité, le plus récent gagne
    const int u = e->tampon[(e->i_tampon - 1 - i + 2 * FUSION_V_STAB) % FUSION_V_STAB];
    if (u >= 1 && u <= 5) {
      compte[u]++;
      if (compte[u] > compte[best]) best = u;
    }
  }
  s->etat = best == 0 ? FUSION_ETAT_NON_CLASSEE : FUSION_ETAT_ALERTE;
  s->type = (int8_t)best;
}

#ifdef FUSION_VERIFIER_VECTEURS
#include <stdio.h>

#include "fusion_test_vectors.h"

int fusion_verifier_vecteurs_test() {
  int ecarts = 0;
  FusionEtat e;
  FusionSortie s;
  for (int n = 0; n < FUSION_TV_N_PAS; ++n) {
    if (FUSION_TV_RESET[n]) fusion_reinitialiser(&e);
    fusion_pas(&e, FUSION_TV_SCORE[n], FUSION_TV_DEPASSE[n] != 0, FUSION_TV_LOGITS[n], &s);
    if (s.etat != FUSION_TV_ETAT[n] || s.type != FUSION_TV_TYPE[n]) {
      ++ecarts;
      printf("pas %d : état %d type %d, attendu %d %d\n", n, s.etat, s.type, FUSION_TV_ETAT[n], FUSION_TV_TYPE[n]);
    }
  }
  return ecarts;
}
#endif
