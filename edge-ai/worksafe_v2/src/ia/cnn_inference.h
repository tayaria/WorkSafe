// cnn_inference.h — classification du type d'anomalie (CNN 1D WorkSafe) avec TensorFlow Lite Micro.
// Généré par cnn/export_tflite.ipynb. Code de référence NON COMPILÉ sur carte : voir LISEZMOI.md.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "cnn_config.h"

struct CnnResultat {
  float probas[CNN_N_CLASSES];  // softmax de la fenêtre courante
  int classe_fenetre;           // argmax de la fenêtre courante (sans vote)
  int classe_vote;              // classe après le vote sur les CNN_K_VOTE dernières fenêtres
  float confiance;              // probabilité moyenne de classe_vote sur le vote
  int classe;                   // décision finale : classe_vote, ou CNN_INCERTAIN si l'option est active
};

// Charge le modèle et alloue l'arena. Renvoie false (et écrit un message) en cas d'erreur.
bool cnn_init();
// À appeler au début de chaque séance : vide le tampon du vote (le vote ne traverse jamais deux séances).
void cnn_nouvelle_seance();
// Traite une fenêtre DÉJÀ NORMALISÉE de CNN_W x CNN_N_CANAUX valeurs (ordre des canaux de cnn_config.h).
bool cnn_traiter_fenetre(const float z[CNN_W][CNN_N_CANAUX], CnnResultat* res);
// Octets d'arena réellement utilisés (après cnn_init) : pour ajuster CNN_ARENA_SIZE.
size_t cnn_arena_utilisee();

// ---- Fonctions de calcul pur (sans TFLite Micro), utilisées aussi par les vecteurs de test ----
void cnn_quantifier_entree(const float z[CNN_W][CNN_N_CANAUX], int8_t q[CNN_N_VALEURS]);
void cnn_softmax(const int8_t sortie_q[CNN_N_CLASSES], float probas[CNN_N_CLASSES]);
// Seuil de confiance : CNN_INCERTAIN si l'option est active et confiance < tau, sinon classe_vote.
// (cnn_traiter_fenetre l'appelle avec CNN_TAU et CNN_UTILISER_INCERTAIN.)
int cnn_appliquer_seuil(int classe_vote, float confiance, float tau, bool actif);

struct CnnVote {
  float tampon[CNN_K_VOTE][CNN_N_CLASSES];
  int n;      // fenêtres dans le tampon (<= CNN_K_VOTE)
  int debut;  // indice de la plus ancienne
};
void cnn_vote_reinitialiser(CnnVote* v);
// Ajoute les probabilités de la fenêtre courante ; renvoie la classe votée et sa confiance.
int cnn_vote_ajouter(CnnVote* v, const float probas[CNN_N_CLASSES], float* confiance);

#ifdef CNN_VERIFIER_VECTEURS
// Compare le firmware aux vecteurs de cnn_test_vectors.h. Renvoie le nombre d'échecs (0 = OK).
int cnn_verifier_vecteurs_test();
#endif
