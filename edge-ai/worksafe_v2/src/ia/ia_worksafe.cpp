// ia_worksafe.cpp — intégration de l'IA dans worksafe_v2.ino (voir ia_worksafe.h). MODE OMBRE : rien n'est commandé.
//
// Chaîne pour chaque fenêtre (toutes les 0,5 s pendant la course) :
//   50 x 6 mesures brutes à 50 Hz -> ae_preparer_entree (normalisation, écrêtage ±8, quantification int8, UNE fois)
//   -> le même tableau int8 entre dans les DEUX modèles -> score de l'AE (ae_calculer_score) -> logits du CNN
//   -> fusion_pas (règle gelée de fusion_config.h) -> état OK / ALERTE(type) / ALERTE non classée.
// Les fonctions de calcul viennent de firmware_pack/ (copies identiques, compilées avec -DAE_SANS_TFLM
// -DCNN_SANS_TFLM via build_opt.h) ; ce fichier crée lui-même les deux interpréteurs TFLite Micro.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")  // pas de multiplication-addition fusionnée (voir RAPPORT_INTEGRATION.md)
#endif
#include "ia_worksafe.h"

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <math.h>
#include <string.h>

#include "ae_inference.h"
#include "ae_model_data.h"
#include "cnn_inference.h"
#include "cnn_model_data.h"
#include "fusion_logic.h"
#include "ia_version.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if !defined(AE_SANS_TFLM) || !defined(CNN_SANS_TFLM)
#error "build_opt.h doit définir AE_SANS_TFLM et CNN_SANS_TFLM (les interpréteurs sont créés dans ia_worksafe.cpp)"
#endif

// ---- Réglages internes ----
#define IA_ARENA_AE (16 * 1024)      // taille de départ ; la commande I affiche l'usage réel (arena_used_bytes)
#define IA_ARENA_CNN (16 * 1024)
#define IA_PILE_TACHE (16 * 1024)    // octets ; la commande I affiche la marge restante
#define IA_PRIORITE_TACHE 1          // basse (au-dessus de la tâche inactive)
#define IA_COEUR_TACHE 0             // loop() tourne sur le cœur 1 : l'IA calcule sur le cœur 0
#define IA_SERIE_TX 2048             // tampon d'émission série (le journal L ne bloque jamais la boucle)

namespace {
// ---- Modèles ----
alignas(16) uint8_t g_arena_ae[IA_ARENA_AE];
alignas(16) uint8_t g_arena_cnn[IA_ARENA_CNN];
tflite::MicroInterpreter* g_ae = nullptr;
tflite::MicroInterpreter* g_cnn = nullptr;
IaOptions g_opt;
bool g_init_ok = false;
bool g_cuisse = true;
volatile bool g_active = false;

// ---- Mesures et fenêtres (contexte de loop()) ----
IaDecim g_decim;
IaFenetres g_fen;
IaPlausibilite g_plaus;
uint32_t g_indice = 0;               // indice du prochain échantillon à 50 Hz
bool g_journal = false;              // commande L
uint32_t g_lignes_perdues = 0;       // lignes du journal non écrites (tampon série plein)

// ---- Échange avec la tâche IA ----
struct Resultat {
  uint32_t seq;                      // incrémenté à chaque fenêtre calculée
  uint32_t fenetre;                  // indice de la fenêtre
  uint32_t dernier;                  // indice du dernier échantillon de la fenêtre
  float score;
  bool depasse;
  int8_t classe;
  float pmax, pfatigue;
  uint8_t etat;
  int8_t type;
  uint32_t us_prep, us_ae, us_cnn, us_fusion, us_total;
};
portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
float g_entree[IA_W][IA_N_CANAUX];  // fenêtre donnée à la tâche (écrite seulement quand la tâche est libre)
uint32_t g_entree_fenetre = 0, g_entree_dernier = 0;
volatile bool g_occupee = false;     // la tâche calcule (ou exécute T / B)
volatile uint32_t g_reset_seq = 0;   // demandes de remise à zéro de la fusion
volatile char g_outil = 0;           // 'T' ou 'B' demandé à la tâche
Resultat g_res = {};                 // dernier résultat (protégé par g_mux)
uint32_t g_res_affiche = 0;          // dernier seq écrit dans le journal
uint32_t g_fenetres = 0, g_perdues = 0, g_us_max = 0;
uint64_t g_us_somme = 0;
TaskHandle_t g_tache = nullptr;
StaticTask_t g_tache_tcb;
StackType_t g_tache_pile[IA_PILE_TACHE / sizeof(StackType_t)];

// ---- État propre à la tâche (ou à loop() si AI_SEPARATE_TASK = 0) ----
FusionEtat g_fusion;
uint32_t g_reset_applique = 0;
float g_z[AE_N_VALEURS];

inline uint32_t us_depuis(uint32_t t0) { return (uint32_t)micros() - t0; }

bool verifier_modele(tflite::MicroInterpreter* it, const char* nom, size_t n_sortie, float in_scale, int in_zp,
                     float out_scale, int out_zp) {
  const TfLiteTensor* e = it->input(0);
  const TfLiteTensor* s = it->output(0);
  const bool ok = e->type == kTfLiteInt8 && s->type == kTfLiteInt8 && e->bytes == AE_N_VALEURS && s->bytes == n_sortie &&
                  e->params.zero_point == in_zp && s->params.zero_point == out_zp &&
                  fabsf(e->params.scale - in_scale) <= 1e-6f * in_scale && fabsf(s->params.scale - out_scale) <= 1e-6f * out_scale;
  if (!ok) Serial.printf("# IA ERREUR : le modèle %s ne correspond pas à sa configuration (types, tailles ou quantification)\n", nom);
  return ok;
}

bool creer_interpreteurs() {
  static tflite::MicroMutableOpResolver<3> res_ae;    // seulement les opérations des deux modèles
  static tflite::MicroMutableOpResolver<3> res_cnn;
  if (res_ae.AddConv2D() != kTfLiteOk || res_ae.AddFullyConnected() != kTfLiteOk || res_ae.AddReshape() != kTfLiteOk ||
      res_cnn.AddConv2D() != kTfLiteOk || res_cnn.AddFullyConnected() != kTfLiteOk || res_cnn.AddReshape() != kTfLiteOk) {
    Serial.println("# IA ERREUR : résolveur d'opérations");
    return false;
  }
  const tflite::Model* m_ae = tflite::GetModel(ae_model_data);
  const tflite::Model* m_cnn = tflite::GetModel(cnn_model_data);
  if (m_ae->version() != TFLITE_SCHEMA_VERSION || m_cnn->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("# IA ERREUR : version de schéma des modèles %d/%d, attendue %d\n", (int)m_ae->version(),
                  (int)m_cnn->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  static tflite::MicroInterpreter it_ae(m_ae, res_ae, g_arena_ae, IA_ARENA_AE);
  static tflite::MicroInterpreter it_cnn(m_cnn, res_cnn, g_arena_cnn, IA_ARENA_CNN);
  if (it_ae.AllocateTensors() != kTfLiteOk) {
    Serial.printf("# IA ERREUR : AllocateTensors de l'autoencoder a échoué (arena de %d octets trop petite ?)\n", IA_ARENA_AE);
    return false;
  }
  if (it_cnn.AllocateTensors() != kTfLiteOk) {
    Serial.printf("# IA ERREUR : AllocateTensors du CNN a échoué (arena de %d octets trop petite ?)\n", IA_ARENA_CNN);
    return false;
  }
  if (!verifier_modele(&it_ae, "autoencoder", AE_N_VALEURS, AE_IN_SCALE, AE_IN_ZERO_POINT, AE_OUT_SCALE, AE_OUT_ZERO_POINT) ||
      !verifier_modele(&it_cnn, "CNN", CNN_N_CLASSES, CNN_IN_SCALE, CNN_IN_ZERO_POINT, CNN_OUT_SCALE, CNN_OUT_ZERO_POINT))
    return false;
  // Une seule préparation de l'entrée pour les deux modèles : seulement si leurs quantifications d'entrée sont identiques.
  const TfLiteTensor* ea = it_ae.input(0);
  const TfLiteTensor* ec = it_cnn.input(0);
  if (ea->params.zero_point != ec->params.zero_point || ea->params.scale != ec->params.scale || AE_CLIP != CNN_CLIP) {
    Serial.printf("# IA ERREUR : quantifications d'entrée différentes (AE %.7f/%d, CNN %.7f/%d) : un seul tableau int8 "
                  "ne peut pas servir aux deux modèles\n", (double)ea->params.scale, (int)ea->params.zero_point,
                  (double)ec->params.scale, (int)ec->params.zero_point);
    return false;
  }
  g_ae = &it_ae;
  g_cnn = &it_cnn;
  return true;
}

// Calcule une fenêtre (contexte de la tâche IA, ou de loop() sans tâche).
void inferer(const float brut[IA_W][IA_N_CANAUX], Resultat* r) {
  if (g_reset_applique != g_reset_seq) {           // remise à zéro demandée par calibrate()
    g_reset_applique = g_reset_seq;
    fusion_reinitialiser(&g_fusion);
  }
  const uint32_t t0 = micros();
  int8_t* q = g_ae->input(0)->data.int8;
  ae_preparer_entree(brut, g_z, q);               // normalisation + écrêtage + quantification, une seule fois
  memcpy(g_cnn->input(0)->data.int8, q, AE_N_VALEURS);
  const uint32_t t1 = micros();
  const bool ok_ae = g_ae->Invoke() == kTfLiteOk;
  const float score = ok_ae ? ae_calculer_score(g_z, g_ae->output(0)->data.int8) : NAN;
  const uint32_t t2 = micros();
  const bool ok_cnn = g_cnn->Invoke() == kTfLiteOk;
  const uint32_t t3 = micros();
  FusionSortie s = {};
  float p[CNN_N_CLASSES] = {0};
  if (ok_ae && ok_cnn) {
    fusion_pas(&g_fusion, score, score >= AE_SEUIL_COMMUN, g_cnn->output(0)->data.int8, &s);
    cnn_softmax(g_cnn->output(0)->data.int8, p);   // probabilité de fatigue, pour le journal seulement
  }
  const uint32_t t4 = micros();
  r->score = score;
  r->depasse = ok_ae && score >= AE_SEUIL_COMMUN;
  r->classe = (ok_ae && ok_cnn) ? s.classe_cnn : -1;
  r->pmax = s.proba_cnn;
  r->pfatigue = p[1];
  r->etat = (ok_ae && ok_cnn) ? s.etat : 255;      // 255 = erreur d'inférence
  r->type = s.type;
  r->us_prep = t1 - t0;
  r->us_ae = t2 - t1;
  r->us_cnn = t3 - t2;
  r->us_fusion = t4 - t3;
  r->us_total = t4 - t0;
}

void publier(const Resultat& r) {
  portENTER_CRITICAL(&g_mux);
  const uint32_t seq = g_res.seq + 1;
  g_res = r;
  g_res.seq = seq;
  g_fenetres++;
  g_us_somme += r.us_total;
  if (r.us_total > g_us_max) g_us_max = r.us_total;
  portEXIT_CRITICAL(&g_mux);
}

void banc_de_mesure();

void tache_ia(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);       // attend une fenêtre (ou une commande T / B)
    const char outil = g_outil;
    if (outil == 'T') {
      if (g_opt.autotest) g_opt.autotest(true);
      else Serial.println("# IA auto-test indisponible (compilé avec AI_AUTOTEST_BOOT = 0 : vecteurs de test absents)");
    } else if (outil == 'B') {
      banc_de_mesure();
    } else {
      Resultat r = {};
      r.fenetre = g_entree_fenetre;
      r.dernier = g_entree_dernier;
      inferer(g_entree, &r);
      publier(r);
    }
    g_outil = 0;
    g_occupee = false;
  }
}

// Envoie une fenêtre prête à la tâche (ou la calcule directement). Ne bloque jamais avec la tâche.
void envoyer_fenetre(uint32_t dernier) {
  const uint32_t indice = g_fen.fenetres;
  if (!g_opt.tache_separee) {
    float brut[IA_W][IA_N_CANAUX];
    ia_fenetres_extraire(&g_fen, brut);
    Resultat r = {};
    r.fenetre = indice;
    r.dernier = dernier;
    inferer(brut, &r);
    publier(r);
    return;
  }
  if (g_occupee) {                                 // la précédente n'est pas finie : on ignore celle-ci
    g_perdues++;
    return;
  }
  ia_fenetres_extraire(&g_fen, g_entree);          // copie avant calcul
  g_entree_fenetre = indice;
  g_entree_dernier = dernier;
  g_occupee = true;
  xTaskNotifyGive(g_tache);
}

// Lance T ou B : dans la tâche si elle existe (la boucle n'attend pas), sinon directement.
void lancer_outil(char c) {
  if (!g_init_ok) {
    Serial.println("# IA indisponible (initialisation échouée)");
    return;
  }
  if (!g_opt.tache_separee || g_tache == nullptr) {
    if (c == 'T') {
      if (g_opt.autotest) g_opt.autotest(true);
      else Serial.println("# IA auto-test indisponible (compilé avec AI_AUTOTEST_BOOT = 0 : vecteurs de test absents)");
    } else {
      banc_de_mesure();
    }
    return;
  }
  if (g_occupee) {
    Serial.println("# IA occupée : recommencer dans une seconde");
    return;
  }
  g_outil = c;
  g_occupee = true;
  xTaskNotifyGive(g_tache);
}

void ecrire_ligne(const char* l, int n) {          // journal : jamais bloquant
  if (Serial.availableForWrite() >= n) Serial.write((const uint8_t*)l, n);
  else g_lignes_perdues++;
}

void journal_echantillon(const float e[IA_N_CANAUX], bool course) {
  char l[112];
  const int n = snprintf(l, sizeof(l), "%lu,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d\n", (unsigned long)g_indice, (double)e[0],
                         (double)e[1], (double)e[2], (double)e[3], (double)e[4], (double)e[5], course ? 1 : 0);
  ecrire_ligne(l, n);
}

void journal_resultat() {
  Resultat r;
  portENTER_CRITICAL(&g_mux);
  r = g_res;
  portEXIT_CRITICAL(&g_mux);
  if (r.seq == g_res_affiche) return;
  g_res_affiche = r.seq;
  if (!g_journal) return;
  char l[112];
  const int n = snprintf(l, sizeof(l), "#IA,%lu,%lu,%.5f,%d,%d,%.4f,%d,%d,%lu\n", (unsigned long)r.fenetre,
                         (unsigned long)r.dernier, (double)r.score, r.depasse ? 1 : 0, (int)r.classe, (double)r.pmax,
                         (int)r.etat, (int)r.type, (unsigned long)r.us_total);
  ecrire_ligne(l, n);
}

const char* aide_canal(int c, const IaStatsCanal& s) {
  if (s.rapport_std < 0.3f) return "trop peu de mouvement : refaire P pendant la course (ou la marche)";
  switch (c) {
    case 0: return "vérifier l'orientation du capteur tibia (axe x le long du tibia, environ +1 g debout) et l'unité (g)";
    case 1: return "vérifier l'orientation du capteur tibia (axe y avant/arrière) et l'unité (g)";
    case 2: return "vérifier l'orientation du capteur tibia (axe z latéral) et l'unité (g)";
    case 3: return "vérifier l'axe et le sens du gyroscope du tibia (°/s)";
    case 4: return "vérifier l'axe et le sens du gyroscope de la cuisse (°/s)";
    default: return "vérifier le signe de KNEE_SIGN (genou fléchi = angle positif) et la calibration jambe tendue";
  }
}

void plausibilite() {
  IaStatsCanal st[IA_N_CANAUX];
  uint32_t n_course = 0;
  const uint32_t n = ia_plaus_calculer(&g_plaus, AE_MEAN, AE_STD, st, &n_course);
  Serial.printf("# P plausibilité sur les %lu dernières mesures à 50 Hz (%.0f s, dont %.0f s de course)\n",
                (unsigned long)n, n / 50.0, n_course / 50.0);
  if (n < 500) {
    Serial.println("# P pas assez de mesures (moins de 10 s) : marcher ou courir puis recommencer");
    return;
  }
  int suspects = 0;
  for (int c = 0; c < IA_N_CANAUX; ++c) {
    const IaStatsCanal& s = st[c];
    Serial.printf("# P %-11s moy %9.3f (app %9.3f, écart %.1f ét) ét %8.3f (app %8.3f, x%.2f) min %9.3f max %9.3f -> %s\n",
                  AE_CANAUX[c], (double)s.moyenne, (double)AE_MEAN[c], (double)s.ecart_moyenne_en_std, (double)s.ecart_type,
                  (double)AE_STD[c], (double)s.rapport_std, (double)s.mini, (double)s.maxi, s.suspect ? "SUSPECT" : "OK");
    if (s.suspect) {
      Serial.printf("# P   aide %s : %s\n", AE_CANAUX[c], aide_canal(c, s));
      suspects++;
    }
  }
  Serial.printf("# P résultat : %s (%d canal(aux) suspect(s))\n", suspects ? "SUSPECT" : "OK", suspects);
}

void informations() {
  uint32_t fen, perdues, us_max;
  uint64_t somme;
  portENTER_CRITICAL(&g_mux);
  fen = g_fenetres;
  perdues = g_perdues;
  us_max = g_us_max;
  somme = g_us_somme;
  portEXIT_CRITICAL(&g_mux);
  Serial.printf("# I pack : ae_int8.tflite %s, cnn_int8.tflite %s\n", IA_SHA_AE, IA_SHA_CNN);
  Serial.printf("# I IA : %s ; tas libre %u octets (minimum atteint %u) ; CPU %u MHz\n",
                g_active ? "active (mode ombre)" : (g_init_ok ? "désactivée (cuisse absente)" : "désactivée (initialisation échouée)"),
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMinFreeHeap(), (unsigned)getCpuFrequencyMhz());
  if (g_init_ok)
    Serial.printf("# I arena : autoencoder %u / %d octets, CNN %u / %d octets\n", (unsigned)g_ae->arena_used_bytes(),
                  IA_ARENA_AE, (unsigned)g_cnn->arena_used_bytes(), IA_ARENA_CNN);
  if (g_tache)
    Serial.printf("# I tâche IA : pile %d octets, marge minimale %u octets (uxTaskGetStackHighWaterMark)\n", IA_PILE_TACHE,
                  (unsigned)uxTaskGetStackHighWaterMark(g_tache));
  Serial.printf("# I fenêtres calculées %lu, perdues %lu ; inférence moyenne %.2f ms, maximum %.2f ms ; lignes de journal perdues %lu\n",
                (unsigned long)fen, (unsigned long)perdues, fen ? (double)somme / fen / 1000.0 : 0.0, us_max / 1000.0,
                (unsigned long)g_lignes_perdues);
  Serial.printf("# I options : tâche séparée %d, décimation %s, champs série %d, champs BLE %d, auto-test au démarrage %d, "
                "vecteurs de test %s, journal L %s\n", g_opt.tache_separee, g_opt.decim_mode ? "moyenne des 4" : "une sur 4",
                g_opt.afficher_serie, g_opt.afficher_ble, g_opt.autotest_boot, g_opt.autotest ? "compilés" : "absents",
                g_journal ? "actif" : "inactif");
}

struct Chrono {
  uint32_t n = 0, mini = UINT32_MAX, maxi = 0;
  uint64_t somme = 0;
  void ajouter(uint32_t us) {
    n++;
    somme += us;
    if (us < mini) mini = us;
    if (us > maxi) maxi = us;
  }
  double moyenne() const { return n ? (double)somme / n : 0.0; }
};

void banc_de_mesure() {
  static float brut[IA_W][IA_N_CANAUX];
  if (g_opt.fenetre_test) memcpy(brut, g_opt.fenetre_test, sizeof(brut));
  else memset(brut, 0, sizeof(brut));
  Chrono prep, ae, cnn, fus;
  FusionEtat etat;                                  // état local : le banc ne touche pas la fusion en cours
  fusion_reinitialiser(&etat);
  FusionSortie s;
  static float z[AE_N_VALEURS];
  for (int i = 0; i < 200; ++i) {
    uint32_t t = micros();
    ae_preparer_entree(brut, z, g_ae->input(0)->data.int8);
    memcpy(g_cnn->input(0)->data.int8, g_ae->input(0)->data.int8, AE_N_VALEURS);
    prep.ajouter(us_depuis(t));
    t = micros();
    g_ae->Invoke();
    const float score = ae_calculer_score(z, g_ae->output(0)->data.int8);
    ae.ajouter(us_depuis(t));
    t = micros();
    g_cnn->Invoke();
    cnn.ajouter(us_depuis(t));
    t = micros();
    fusion_pas(&etat, score, score >= AE_SEUIL_COMMUN, g_cnn->output(0)->data.int8, &s);
    fus.ajouter(us_depuis(t));
  }
  const double total = prep.moyenne() + ae.moyenne() + cnn.moyenne() + fus.moyenne();
  Serial.printf("# B banc de mesure : 200 inférences, CPU %u MHz, %s\n", (unsigned)getCpuFrequencyMhz(),
                g_opt.fenetre_test ? "fenêtre de test" : "fenêtre nulle");
  const Chrono* c[4] = {&prep, &ae, &cnn, &fus};
  const char* noms[4] = {"préparation", "autoencoder", "CNN", "fusion"};
  for (int k = 0; k < 4; ++k)
    Serial.printf("# B %-12s moyenne %8.1f µs, minimum %6lu µs, maximum %6lu µs\n", noms[k], c[k]->moyenne(),
                  (unsigned long)c[k]->mini, (unsigned long)c[k]->maxi);
  Serial.printf("# B total par fenêtre %.2f ms ; une fenêtre toutes les 0,5 s -> charge estimée %.2f %% du temps d'un cœur\n",
                total / 1000.0, 100.0 * total / 500000.0);
}
}  // namespace

// ===========================================================================
// API
// ===========================================================================
bool ia_setup(const IaOptions& options) {
  g_opt = options;
  ia_decim_init(&g_decim, g_opt.decim_mode);
  ia_fenetres_vider(&g_fen);
  ia_plaus_vider(&g_plaus);
  fusion_reinitialiser(&g_fusion);
  const uint32_t tas_avant = ESP.getFreeHeap();
  g_init_ok = creer_interpreteurs();
  if (!g_init_ok) {
    Serial.println("# IA désactivée : initialisation échouée (le firmware continue avec ses règles habituelles)");
    return false;
  }
  Serial.printf("# IA prête (mode ombre) : arena autoencoder %u o, CNN %u o, tas libre %u o (avant : %u o)\n",
                (unsigned)g_ae->arena_used_bytes(), (unsigned)g_cnn->arena_used_bytes(), (unsigned)ESP.getFreeHeap(),
                (unsigned)tas_avant);
  if (g_opt.autotest_boot && g_opt.autotest) g_opt.autotest(true);
  if (g_opt.tache_separee) {
    g_tache = xTaskCreateStaticPinnedToCore(tache_ia, "ia", IA_PILE_TACHE / sizeof(StackType_t), nullptr,
                                            IA_PRIORITE_TACHE, g_tache_pile, &g_tache_tcb, IA_COEUR_TACHE);
    if (g_tache == nullptr) {
      Serial.println("# IA désactivée : création de la tâche impossible");
      g_init_ok = false;
      return false;
    }
  }
  g_active = true;
  return true;
}

void ia_capteur_cuisse(bool present) {
  g_cuisse = present;
  if (!present) {
    g_active = false;
    Serial.println("# IA désactivée (cuisse absente)");
  }
}

void ia_reset_session() {
  ia_fenetres_vider(&g_fen);
  g_fen.fenetres = 0;
  g_reset_seq = g_reset_seq + 1;                  // appliqué par la tâche avant la prochaine fenêtre
  if (!g_opt.tache_separee) {
    g_reset_applique = g_reset_seq;
    fusion_reinitialiser(&g_fusion);
  }
  portENTER_CRITICAL(&g_mux);
  g_fenetres = g_perdues = g_us_max = 0;
  g_us_somme = 0;
  portEXIT_CRITICAL(&g_mux);
  if (g_journal) Serial.println("#RESET");
}

int ia_mesure_200hz(uint32_t t_us, const float v[IA_N_CANAUX], bool course, bool calibration,
                    float e50[IA_MAX_ECHANTILLONS][IA_N_CANAUX]) {
  const int k = ia_decim_ajouter(&g_decim, t_us, v, e50, IA_MAX_ECHANTILLONS);
  const bool actif = g_active && course && g_cuisse && !calibration;
  for (int i = 0; i < k; ++i) {
    if (g_cuisse) ia_plaus_ajouter(&g_plaus, e50[i], course);
    if (g_journal) journal_echantillon(e50[i], course);
    if (ia_fenetres_ajouter(&g_fen, e50[i], actif)) envoyer_fenetre(g_indice);
    g_indice++;
  }
  journal_resultat();
  return k;
}

void ia_commandes() {
  while (Serial.available() > 0) {
    const int c = Serial.read();
    switch (c) {
      case 'T': case 't': lancer_outil('T'); break;
      case 'B': case 'b': lancer_outil('B'); break;
      case 'I': case 'i': informations(); break;
      case 'P': case 'p': plausibilite(); break;
      case 'L': case 'l':
        g_journal = !g_journal;
        if (g_journal)
          Serial.println("# journal IA actif : indice,ax_tibia,ay_tibia,az_tibia,gy_tibia,gy_cuisse,angle_genou,course | "
                         "#IA,fenetre,dernier_indice,score_ae,alerte_ae,classe_cnn,pmax,etat_fusion,type,us_inference");
        else
          Serial.println("# journal IA arrêté");
        break;
      default: break;
    }
  }
}

int ia_champs_trame(char* buf, size_t taille) {
  Resultat r;
  portENTER_CRITICAL(&g_mux);
  r = g_res;
  portEXIT_CRITICAL(&g_mux);
  if (r.seq == 0) return snprintf(buf, taille, ",A:-,Y:-,Z:-,I:-");
  return snprintf(buf, taille, ",A:%.2f,Y:%d,Z:%d,I:%.1f", (double)r.score, (int)r.classe, (int)r.etat, r.us_total / 1000.0);
}

uint8_t ia_etat_fusion(int8_t* type) {
  portENTER_CRITICAL(&g_mux);
  const uint8_t e = g_res.etat;
  if (type) *type = g_res.type;
  portEXIT_CRITICAL(&g_mux);
  return e;
}

bool ia_active() { return g_active; }

// Appelée par ia_autotest.cpp
tflite::MicroInterpreter* ia_interpreteur_ae() { return g_ae; }
tflite::MicroInterpreter* ia_interpreteur_cnn() { return g_cnn; }
void ia_serie_tx_taille() { Serial.setTxBufferSize(IA_SERIE_TX); }
