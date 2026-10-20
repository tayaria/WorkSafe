// ia_autotest.cpp — auto-test de l'IA sur la carte : rejoue les vecteurs de test de firmware_pack/
// (ae_test_vectors.h, cnn_test_vectors.h, fusion_test_vectors.h, copiés à l'identique).
//   - autoencoder : entrée int8 (normalisation + écrêtage + quantification) exacte, sortie int8 du modèle (Invoke)
//     exacte, score S3 à 1e-5 relatif près, dépassement du seuil exact ;
//   - CNN : entrée int8 exacte, sortie int8 (Invoke) exacte, probabilités à 1e-5 relatif près, classe, vote et
//     seuil de confiance exacts ;
//   - fusion : état et type exacts à tous les pas des séquences.
// Tolérances identiques à firmware_pack/pc_tests. Ce fichier n'est lié au programme que si le sketch référence
// ia_autotest (AI_AUTOTEST_BOOT = 1) : sinon l'éditeur de liens retire le code ET les vecteurs (économie de flash).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif
#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "ae_inference.h"
#include "ae_test_vectors.h"
#include "cnn_inference.h"
#include "cnn_test_vectors.h"
#include "fusion_logic.h"
#include "fusion_test_vectors.h"
#include "ia_worksafe.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

tflite::MicroInterpreter* ia_interpreteur_ae();
tflite::MicroInterpreter* ia_interpreteur_cnn();

namespace {
#define TOL_REL 1e-5f
#define TOL_ABS 1e-9f
#define MAX_DETAILS 10

struct Compteur {
  long n = 0, ecarts = 0;
  int details = 0;
  bool afficher = true;
  void entier(const char* quoi, int vecteur, int i, long obtenu, long attendu) {
    ++n;
    if (obtenu == attendu) return;
    ++ecarts;
    if (afficher && details++ < MAX_DETAILS)
      Serial.printf("#   écart %s vecteur %d indice %d : obtenu %ld, attendu %ld\n", quoi, vecteur, i, obtenu, attendu);
  }
  void flottant(const char* quoi, int vecteur, int i, float obtenu, float attendu) {
    ++n;
    if (fabsf(obtenu - attendu) <= TOL_REL * fabsf(attendu) + TOL_ABS) return;
    ++ecarts;
    if (afficher && details++ < MAX_DETAILS)
      Serial.printf("#   écart %s vecteur %d indice %d : obtenu %.9g, attendu %.9g\n", quoi, vecteur, i, (double)obtenu,
                    (double)attendu);
  }
};
}  // namespace

const float (*ia_fenetre_test())[IA_N_CANAUX] { return AE_TV_BRUT[0]; }

int ia_autotest(bool afficher) {
  tflite::MicroInterpreter* ae = ia_interpreteur_ae();
  tflite::MicroInterpreter* cnn = ia_interpreteur_cnn();
  if (ae == nullptr || cnn == nullptr) {
    Serial.println("# IA auto-test : impossible (interpréteurs non initialisés)");
    return -1;
  }
  const uint32_t t0 = millis();
  Compteur c;
  c.afficher = afficher;
  // ---- autoencoder ----
  static float z[AE_N_VALEURS];
  for (int n = 0; n < AE_N_VECTEURS; ++n) {
    int8_t* q = ae->input(0)->data.int8;
    ae_preparer_entree(AE_TV_BRUT[n], z, q);
    for (int i = 0; i < AE_N_VALEURS; ++i) c.entier("ae entrée int8", n, i, q[i], AE_TV_ENTREE_Q[n][i]);
    if (ae->Invoke() != kTfLiteOk) {
      c.entier("ae Invoke", n, 0, 1, 0);
      continue;
    }
    const int8_t* s = ae->output(0)->data.int8;
    for (int i = 0; i < AE_N_VALEURS; ++i) c.entier("ae sortie int8", n, i, s[i], AE_TV_SORTIE_Q[n][i]);
    const float score = ae_calculer_score(z, s);
    c.flottant("ae score", n, 0, score, AE_TV_SCORE[n]);
    c.entier("ae dépassement", n, 0, score >= AE_SEUIL_COMMUN, AE_TV_SCORE[n] >= AE_SEUIL_COMMUN);
  }
  const long n_ae = c.n, e_ae = c.ecarts;
  // ---- CNN ----
  CnnVote vote;
  cnn_vote_reinitialiser(&vote);
  for (int n = 0; n < CNN_N_VECTEURS; ++n) {
    int8_t* q = cnn->input(0)->data.int8;
    cnn_quantifier_entree(CNN_TV_Z[n], q);
    for (int i = 0; i < CNN_N_VALEURS; ++i) c.entier("cnn entrée int8", n, i, q[i], CNN_TV_ENTREE_Q[n][i]);
    if (cnn->Invoke() != kTfLiteOk) {
      c.entier("cnn Invoke", n, 0, 1, 0);
      continue;
    }
    const int8_t* s = cnn->output(0)->data.int8;
    for (int k = 0; k < CNN_N_CLASSES; ++k) c.entier("cnn sortie int8", n, k, s[k], CNN_TV_SORTIE_Q[n][k]);
    float p[CNN_N_CLASSES];
    cnn_softmax(s, p);
    int a = 0;
    for (int k = 0; k < CNN_N_CLASSES; ++k) {
      c.flottant("cnn probabilité", n, k, p[k], CNN_TV_PROBAS[n][k]);
      a = p[k] > p[a] ? k : a;
    }
    c.entier("cnn classe", n, 0, a, CNN_TV_CLASSE[n]);
    float conf;
    const int cv = cnn_vote_ajouter(&vote, CNN_TV_PROBAS[n], &conf);
    c.entier("cnn vote", n, 0, cv, CNN_TV_VOTE_CLASSE[n]);
    c.flottant("cnn confiance", n, 0, conf, CNN_TV_VOTE_CONFIANCE[n]);
    const int d = cnn_appliquer_seuil(cv, conf, CNN_TAU, CNN_UTILISER_INCERTAIN != 0);
    c.entier("cnn seuil de confiance", n, 0, d,
             (CNN_UTILISER_INCERTAIN && CNN_TV_VOTE_CONFIANCE[n] < CNN_TAU) ? CNN_INCERTAIN : CNN_TV_VOTE_CLASSE[n]);
  }
  const long n_cnn = c.n - n_ae, e_cnn = c.ecarts - e_ae;
  // ---- fusion (état local : la fusion en cours n'est pas touchée) ----
  FusionEtat e;
  FusionSortie s;
  fusion_reinitialiser(&e);
  for (int n = 0; n < FUSION_TV_N_PAS; ++n) {
    if (FUSION_TV_RESET[n]) fusion_reinitialiser(&e);
    fusion_pas(&e, FUSION_TV_SCORE[n], FUSION_TV_DEPASSE[n] != 0, FUSION_TV_LOGITS[n], &s);
    c.entier("fusion état", n, 0, s.etat, FUSION_TV_ETAT[n]);
    c.entier("fusion type", n, 0, s.type, FUSION_TV_TYPE[n]);
  }
  const long n_fus = c.n - n_ae - n_cnn, e_fus = c.ecarts - e_ae - e_cnn;
  Serial.printf("# IA auto-test détail : autoencoder %ld/%ld, CNN %ld/%ld, fusion %ld/%ld comparaisons justes (%lu ms)\n",
                n_ae - e_ae, n_ae, n_cnn - e_cnn, n_cnn, n_fus - e_fus, n_fus, (unsigned long)(millis() - t0));
  if (c.ecarts == 0)
    Serial.printf("# IA auto-test : OK (%ld comparaisons, 0 écart)\n", c.n);
  else
    Serial.printf("# IA auto-test : ECHEC (%ld comparaisons, %ld écart(s) ; les %d premiers sont listés ci-dessus)\n", c.n,
                  c.ecarts, MAX_DETAILS);
  return (int)c.ecarts;
}
