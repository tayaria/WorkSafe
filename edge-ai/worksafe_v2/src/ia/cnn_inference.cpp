// cnn_inference.cpp — classification du type d'anomalie (CNN 1D WorkSafe) avec TensorFlow Lite Micro.
// Généré par cnn/export_tflite.ipynb.
//
// ATTENTION : code de référence, NON COMPILÉ ni testé sur une carte ESP32. La partie calcul (sans TFLite
// Micro) est compilée et rejouée sur PC (firmware_pack/pc_tests). Points à valider sur la carte :
//   - compilation avec la bibliothèque TFLite Micro utilisée (esp-tflite-micro ou TensorFlowLite_ESP32) ;
//   - CNN_ARENA_SIZE (estimation) : lire cnn_arena_utilisee() et réduire ;
//   - résultats bit à bit avec cnn_test_vectors.h (compiler avec -DCNN_VERIFIER_VECTEURS) ;
//   - temps d'inférence et consommation (voir LISEZMOI.md).
#include "cnn_inference.h"

#include <math.h>
#include <string.h>

// ===========================================================================
// Partie calcul (aucune dépendance à TFLite Micro)
// ===========================================================================
void cnn_quantifier_entree(const float z[CNN_W][CNN_N_CANAUX], int8_t q[CNN_N_VALEURS]) {
  for (int t = 0; t < CNN_W; ++t) {
    for (int c = 0; c < CNN_N_CANAUX; ++c) {
      float x = z[t][c];
      x = x > CNN_CLIP ? CNN_CLIP : (x < -CNN_CLIP ? -CNN_CLIP : x);
      long v = lroundf(x * CNN_IN_INV_SCALE) + CNN_IN_ZERO_POINT;  // arrondi : moitiés loin de zéro
      if (v > 127) v = 127;
      if (v < -128) v = -128;
      q[t * CNN_N_CANAUX + c] = (int8_t)v;  // même ordre que X[t][c] côté Python
    }
  }
}

void cnn_softmax(const int8_t sortie_q[CNN_N_CLASSES], float probas[CNN_N_CLASSES]) {
  float logits[CNN_N_CLASSES];
  for (int c = 0; c < CNN_N_CLASSES; ++c)
    logits[c] = ((float)sortie_q[c] - (float)CNN_OUT_ZERO_POINT) * CNN_OUT_SCALE;  // déquantification
  float m = logits[0];
  for (int c = 1; c < CNN_N_CLASSES; ++c) m = logits[c] > m ? logits[c] : m;
  float s = 0.0f;
  for (int c = 0; c < CNN_N_CLASSES; ++c) {
    probas[c] = expf(logits[c] - m);  // soustraire le maximum évite les débordements
    s += probas[c];
  }
  for (int c = 0; c < CNN_N_CLASSES; ++c) probas[c] = probas[c] / s;
}

int cnn_appliquer_seuil(int classe_vote, float confiance, float tau, bool actif) {
  return (actif && confiance < tau) ? CNN_INCERTAIN : classe_vote;
}

void cnn_vote_reinitialiser(CnnVote* v) {
  v->n = 0;
  v->debut = 0;
}

int cnn_vote_ajouter(CnnVote* v, const float probas[CNN_N_CLASSES], float* confiance) {
  if (v->n < CNN_K_VOTE) {
    memcpy(v->tampon[(v->debut + v->n) % CNN_K_VOTE], probas, sizeof(float) * CNN_N_CLASSES);
    v->n++;
  } else {  // tampon plein : on remplace la plus ancienne
    memcpy(v->tampon[v->debut], probas, sizeof(float) * CNN_N_CLASSES);
    v->debut = (v->debut + 1) % CNN_K_VOTE;
  }
  float somme[CNN_N_CLASSES] = {0};
  int votes[CNN_N_CLASSES] = {0};
  for (int i = 0; i < v->n; ++i) {  // de la plus ancienne à la plus récente
    const float* q = v->tampon[(v->debut + i) % CNN_K_VOTE];
    int a = 0;
    for (int c = 0; c < CNN_N_CLASSES; ++c) {
      somme[c] += q[c];
      if (q[c] > q[a]) a = c;
    }
    votes[a]++;
  }
  int best = 0;
  for (int c = 1; c < CNN_N_CLASSES; ++c) {
#if CNN_REGLE_VOTE == CNN_VOTE_MOYENNE
    if (somme[c] > somme[best]) best = c;
#else  // majorité ; égalité -> plus grande somme de probabilités, puis plus petit indice
    if (votes[c] > votes[best] || (votes[c] == votes[best] && somme[c] > somme[best])) best = c;
#endif
  }
  *confiance = somme[best] / (float)v->n;
  return best;
}

// ===========================================================================
// Partie TFLite Micro
// ===========================================================================
#ifndef CNN_SANS_TFLM
#include "cnn_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
alignas(16) uint8_t g_arena[CNN_ARENA_SIZE];
tflite::MicroInterpreter* g_interp = nullptr;
TfLiteTensor* g_entree = nullptr;
TfLiteTensor* g_sortie = nullptr;
CnnVote g_vote;
}  // namespace

bool cnn_init() {
  const tflite::Model* modele = tflite::GetModel(cnn_model_data);
  if (modele->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("cnn_init : version de schéma %d, attendue %d", (int)modele->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  // Uniquement les opérations présentes dans cnn_int8.tflite : CONV_2D, FULLY_CONNECTED, RESHAPE
  static tflite::MicroMutableOpResolver<3> resolveur;
  if (resolveur.AddConv2D() != kTfLiteOk) return false;
  if (resolveur.AddFullyConnected() != kTfLiteOk) return false;
  if (resolveur.AddReshape() != kTfLiteOk) return false;
  static tflite::MicroInterpreter interpreteur(modele, resolveur, g_arena, CNN_ARENA_SIZE);
  g_interp = &interpreteur;
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("cnn_init : AllocateTensors a échoué, augmenter CNN_ARENA_SIZE");
    g_interp = nullptr;
    return false;
  }
  g_entree = g_interp->input(0);
  g_sortie = g_interp->output(0);
  // Le modèle doit correspondre à cnn_config.h (types, tailles, quantification)
  if (g_entree->type != kTfLiteInt8 || g_sortie->type != kTfLiteInt8 ||
      g_entree->bytes != CNN_N_VALEURS || g_sortie->bytes != CNN_N_CLASSES ||
      g_entree->params.zero_point != CNN_IN_ZERO_POINT || g_sortie->params.zero_point != CNN_OUT_ZERO_POINT ||
      fabsf(g_entree->params.scale - CNN_IN_SCALE) > 1e-6f * CNN_IN_SCALE ||
      fabsf(g_sortie->params.scale - CNN_OUT_SCALE) > 1e-6f * CNN_OUT_SCALE) {
    MicroPrintf("cnn_init : le modèle ne correspond pas à cnn_config.h");
    g_interp = nullptr;
    return false;
  }
  cnn_nouvelle_seance();
  return true;
}

void cnn_nouvelle_seance() { cnn_vote_reinitialiser(&g_vote); }

size_t cnn_arena_utilisee() { return g_interp ? g_interp->arena_used_bytes() : 0; }

bool cnn_traiter_fenetre(const float z[CNN_W][CNN_N_CANAUX], CnnResultat* res) {
  if (g_interp == nullptr || res == nullptr) return false;
  cnn_quantifier_entree(z, g_entree->data.int8);
  if (g_interp->Invoke() != kTfLiteOk) return false;
  cnn_softmax(g_sortie->data.int8, res->probas);
  int a = 0;
  for (int c = 1; c < CNN_N_CLASSES; ++c) a = res->probas[c] > res->probas[a] ? c : a;
  res->classe_fenetre = a;
  res->classe_vote = cnn_vote_ajouter(&g_vote, res->probas, &res->confiance);
  res->classe = cnn_appliquer_seuil(res->classe_vote, res->confiance, CNN_TAU, CNN_UTILISER_INCERTAIN != 0);
  return true;
}

#ifdef CNN_VERIFIER_VECTEURS
#include "cnn_test_vectors.h"

int cnn_verifier_vecteurs_test() {
  int echecs = 0;
  CnnVote vote;
  cnn_vote_reinitialiser(&vote);  // les vecteurs sont aussi enchaînés comme une petite séance
  for (int n = 0; n < CNN_N_VECTEURS; ++n) {
    cnn_quantifier_entree(CNN_TV_Z[n], g_entree->data.int8);
    int diff_entree = 0, diff_sortie = 0;
    for (int i = 0; i < CNN_N_VALEURS; ++i) diff_entree += g_entree->data.int8[i] != CNN_TV_ENTREE_Q[n][i];
    if (g_interp->Invoke() != kTfLiteOk) return -1;
    for (int c = 0; c < CNN_N_CLASSES; ++c) diff_sortie += g_sortie->data.int8[c] != CNN_TV_SORTIE_Q[n][c];
    float p[CNN_N_CLASSES];
    cnn_softmax(g_sortie->data.int8, p);
    float ecart = 0.0f;
    int a = 0;
    for (int c = 0; c < CNN_N_CLASSES; ++c) {
      const float e = fabsf(p[c] - CNN_TV_PROBAS[n][c]);
      ecart = e > ecart ? e : ecart;
      a = p[c] > p[a] ? c : a;
    }
    float confiance;
    const int cv = cnn_vote_ajouter(&vote, CNN_TV_PROBAS[n], &confiance);
    const bool ok = diff_entree == 0 && diff_sortie == 0 && ecart <= CNN_TV_TOLERANCE_PROBA &&
                    a == CNN_TV_CLASSE[n] && cv == CNN_TV_VOTE_CLASSE[n] &&
                    fabsf(confiance - CNN_TV_VOTE_CONFIANCE[n]) <= CNN_TV_TOLERANCE_PROBA;
    MicroPrintf("vecteur %d (%s) : entrée %d diff, sortie %d diff, classe %s (attendue %s), vote %d (attendu %d) -> %s",
                n, CNN_TV_ETIQUETTE[n], diff_entree, diff_sortie, CNN_CLASSES[a], CNN_CLASSES[CNN_TV_CLASSE[n]],
                cv, CNN_TV_VOTE_CLASSE[n], ok ? "OK" : "ÉCHEC");
    echecs += ok ? 0 : 1;
  }
  return echecs;
}
#endif  // CNN_VERIFIER_VECTEURS
#endif  // CNN_SANS_TFLM
