// ae_inference.cpp — inférence de l'autoencoder WorkSafe avec TensorFlow Lite Micro.
// Généré par autoencoder/export_tflite.ipynb.
//
// ATTENTION : code de référence, NON COMPILÉ ni testé sur une carte ESP32 (aucun compilateur
// C++ n'était disponible lors de la génération). Points à valider sur la carte :
//   - compilation avec la bibliothèque TFLite Micro utilisée (esp-tflite-micro ou TensorFlowLite_ESP32) :
//     le constructeur de MicroInterpreter et MicroPrintf ont changé selon les versions ;
//   - AE_ARENA_SIZE (estimation) : lire ae_arena_utilisee() et ajuster ;
//   - résultats bit à bit avec ae_test_vectors.h (compiler avec -DAE_VERIFIER_VECTEURS) ;
//   - temps d'inférence et consommation (voir LISEZMOI.md).
#include "ae_inference.h"

#include <math.h>
#include <string.h>

// ===========================================================================
// Partie calcul (aucune dépendance à TFLite Micro)
// ===========================================================================
static inline float ae_ecreter(float x) {
  return x > AE_CLIP ? AE_CLIP : (x < -AE_CLIP ? -AE_CLIP : x);
}

void ae_preparer_entree(const float brut[AE_W][AE_N_CANAUX], float z[AE_N_VALEURS], int8_t q[AE_N_VALEURS]) {
  for (int t = 0; t < AE_W; ++t) {
    for (int c = 0; c < AE_N_CANAUX; ++c) {
      const int i = t * AE_N_CANAUX + c;  // même ordre que X[t][c] côté Python
      const float zi = ae_ecreter((brut[t][c] - AE_MEAN[c]) * AE_INV_STD[c]);
      z[i] = zi;
      long v = lroundf(zi * AE_IN_INV_SCALE) + AE_IN_ZERO_POINT;  // arrondi : moitiés loin de zéro
      if (v > 127) v = 127;
      if (v < -128) v = -128;
      q[i] = (int8_t)v;
    }
  }
}

static void ae_trier(float* v, int n) {  // tri par insertion, croissant (n <= 120)
  for (int i = 1; i < n; ++i) {
    const float x = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > x) {
      v[j + 1] = v[j];
      --j;
    }
    v[j + 1] = x;
  }
}

float ae_mediane(float* v, int n) {
  ae_trier(v, n);
  return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

float ae_calculer_score(const float z[AE_N_VALEURS], const int8_t sortie_q[AE_N_VALEURS]) {
  static float d2[AE_N_VALEURS];
  for (int i = 0; i < AE_N_VALEURS; ++i) {
    const float r = ((float)sortie_q[i] - (float)AE_OUT_ZERO_POINT) * AE_OUT_SCALE;  // déquantification
    const float d = z[i] - r;
    d2[i] = d * d;
  }
#if AE_SCORE_TYPE == AE_SCORE_S1
  float s = 0.0f;
  for (int i = 0; i < AE_N_VALEURS; ++i) s += d2[i];
  return s / (float)AE_N_VALEURS;
#elif AE_SCORE_TYPE == AE_SCORE_S2
  float e[AE_W];
  for (int t = 0; t < AE_W; ++t) {
    float s = 0.0f;
    for (int c = 0; c < AE_N_CANAUX; ++c) s += d2[t * AE_N_CANAUX + c];
    e[t] = s / (float)AE_N_CANAUX;
  }
  ae_trier(e, AE_W);
  float s = 0.0f;
  for (int k = 0; k < AE_S2_TOP_K; ++k) s += e[AE_W - 1 - k];
  return s / (float)AE_S2_TOP_K;
#else
  float r[AE_N_CANAUX];
  for (int c = 0; c < AE_N_CANAUX; ++c) {
    float s = 0.0f;
    for (int t = 0; t < AE_W; ++t) s += d2[t * AE_N_CANAUX + c];
    r[c] = (s / (float)AE_W) / AE_S3_MEDIANES[c];
  }
#if AE_SCORE_TYPE == AE_SCORE_S3_MOY
  float s = 0.0f;
  for (int c = 0; c < AE_N_CANAUX; ++c) s += r[c];
  return s / (float)AE_N_CANAUX;
#else
  float m = r[0];
  for (int c = 1; c < AE_N_CANAUX; ++c) m = r[c] > m ? r[c] : m;
  return m;
#endif
#endif
}

// ===========================================================================
// Partie TFLite Micro
// ===========================================================================
#ifndef AE_SANS_TFLM
#include "ae_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
alignas(16) uint8_t g_arena[AE_ARENA_SIZE];
tflite::MicroInterpreter* g_interp = nullptr;
TfLiteTensor* g_entree = nullptr;
TfLiteTensor* g_sortie = nullptr;
float g_z[AE_N_VALEURS];

int g_persistance = 0;               // fenêtres consécutives au-dessus du seuil
bool g_calibration_active = false;
int g_n_calib = 0;
float g_calib_scores[AE_N_CALIB];
float g_seuil_athlete = 0.0f;
bool g_calibre = false;
}  // namespace

bool ae_init() {
  const tflite::Model* modele = tflite::GetModel(ae_model_data);
  if (modele->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("ae_init : version de schéma %d, attendue %d", (int)modele->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  // Uniquement les opérations présentes dans ae_int8.tflite : CONV_2D, FULLY_CONNECTED, RESHAPE
  static tflite::MicroMutableOpResolver<3> resolveur;
  if (resolveur.AddConv2D() != kTfLiteOk) return false;
  if (resolveur.AddFullyConnected() != kTfLiteOk) return false;
  if (resolveur.AddReshape() != kTfLiteOk) return false;
  static tflite::MicroInterpreter interpreteur(modele, resolveur, g_arena, AE_ARENA_SIZE);
  g_interp = &interpreteur;
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("ae_init : AllocateTensors a échoué, augmenter AE_ARENA_SIZE");
    g_interp = nullptr;
    return false;
  }
  g_entree = g_interp->input(0);
  g_sortie = g_interp->output(0);
  // Le modèle doit correspondre à ae_config.h (types, tailles, quantification)
  if (g_entree->type != kTfLiteInt8 || g_sortie->type != kTfLiteInt8 ||
      g_entree->bytes != AE_N_VALEURS || g_sortie->bytes != AE_N_VALEURS ||
      g_entree->params.zero_point != AE_IN_ZERO_POINT || g_sortie->params.zero_point != AE_OUT_ZERO_POINT ||
      fabsf(g_entree->params.scale - AE_IN_SCALE) > 1e-6f * AE_IN_SCALE ||
      fabsf(g_sortie->params.scale - AE_OUT_SCALE) > 1e-6f * AE_OUT_SCALE) {
    MicroPrintf("ae_init : le modèle ne correspond pas à ae_config.h");
    g_interp = nullptr;
    return false;
  }
  ae_nouvelle_seance();
  return true;
}

void ae_nouvelle_seance() { g_persistance = 0; }

void ae_demarrer_calibration() {
  g_calibration_active = true;
  g_n_calib = 0;
}

size_t ae_arena_utilisee() { return g_interp ? g_interp->arena_used_bytes() : 0; }

bool ae_traiter_fenetre(const float brut[AE_W][AE_N_CANAUX], AeResultat* res) {
  if (g_interp == nullptr || res == nullptr) return false;
  ae_preparer_entree(brut, g_z, g_entree->data.int8);
  if (g_interp->Invoke() != kTfLiteOk) return false;
  const float score = ae_calculer_score(g_z, g_sortie->data.int8);

  // Calibration : médiane des scores de AE_N_CALIB fenêtres de course normale
  if (g_calibration_active) {
    g_calib_scores[g_n_calib++] = score;
    if (g_n_calib == AE_N_CALIB) {
      g_seuil_athlete = AE_K_CALIB * ae_mediane(g_calib_scores, AE_N_CALIB);
      g_calibre = true;
      g_calibration_active = false;
    }
  }

  const bool calibre = AE_UTILISER_CALIBRATION && g_calibre;
  res->score = score;
  res->seuil = calibre ? g_seuil_athlete : AE_SEUIL_COMMUN;
  res->depasse = score >= res->seuil;
  g_persistance = res->depasse ? g_persistance + 1 : 0;
  res->alerte = g_persistance >= AE_PERSISTANCE;
  res->calibre = calibre;
  return true;
}

#ifdef AE_VERIFIER_VECTEURS
#include "ae_test_vectors.h"

int ae_verifier_vecteurs_test() {
  int echecs = 0;
  static float z[AE_N_VALEURS];
  for (int n = 0; n < AE_N_VECTEURS; ++n) {
    ae_preparer_entree(AE_TV_BRUT[n], z, g_entree->data.int8);
    int diff_entree = 0, diff_sortie = 0;
    for (int i = 0; i < AE_N_VALEURS; ++i) diff_entree += g_entree->data.int8[i] != AE_TV_ENTREE_Q[n][i];
    if (g_interp->Invoke() != kTfLiteOk) return -1;
    for (int i = 0; i < AE_N_VALEURS; ++i) diff_sortie += g_sortie->data.int8[i] != AE_TV_SORTIE_Q[n][i];
    const float score = ae_calculer_score(z, g_sortie->data.int8);
    const float ecart = fabsf(score - AE_TV_SCORE[n]) / AE_TV_SCORE[n];
    const bool ok = diff_entree == 0 && diff_sortie == 0 && ecart <= AE_TV_TOLERANCE_SCORE;
    MicroPrintf("vecteur %d (%s) : entrée %d diff, sortie %d diff, score %f (attendu %f) -> %s", n,
                AE_TV_ETIQUETTE[n], diff_entree, diff_sortie, (double)score, (double)AE_TV_SCORE[n],
                ok ? "OK" : "ÉCHEC");
    echecs += ok ? 0 : 1;
  }
  return echecs;
}
#endif  // AE_VERIFIER_VECTEURS
#endif  // AE_SANS_TFLM
