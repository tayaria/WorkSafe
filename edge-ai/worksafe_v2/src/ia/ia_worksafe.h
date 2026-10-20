// ia_worksafe.h — intégration de l'IA (autoencoder + CNN + fusion) dans worksafe_v2.ino, en MODE OMBRE :
// l'IA calcule et affiche, elle ne commande RIEN (ni LED, ni moteur, ni buzzer, ni niveau).
//
// Dépend d'Arduino (Serial, micros, FreeRTOS) et de TensorFlow Lite Micro (bibliothèque Chirale_TensorFLowLite).
// Les calculs (normalisation, quantification, score, softmax, fusion) sont ceux de firmware_pack/, copiés
// à l'identique dans ce dossier ; ia_core.* contient la logique pure (mesure à 50 Hz, fenêtres, plausibilité).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "ia_core.h"

#define IA_MAX_ECHANTILLONS 12      // échantillons à 50 Hz produits au plus par appel de ia_mesure_200hz

typedef int (*IaAutotestFn)(bool afficher_details);

struct IaOptions {
  bool tache_separee;               // AI_SEPARATE_TASK
  uint8_t decim_mode;               // AI_DECIM_MODE : 1 = moyenne des 4 mesures, 0 = une sur quatre
  bool afficher_serie;              // AI_PRINT_SERIAL : champs A,Y,Z,I ajoutés à la trame série 1 Hz
  bool afficher_ble;                // AI_PRINT_BLE
  bool autotest_boot;               // AI_AUTOTEST_BOOT
  IaAutotestFn autotest;            // ia_autotest si les vecteurs de test sont compilés, sinon nullptr
  const float (*fenetre_test)[IA_N_CANAUX];  // fenêtre BRUTE pour le banc de mesure (nullptr : fenêtre nulle)
};

// Au tout début de setup(), AVANT l'attente du capteur tibia : crée les deux interpréteurs, vérifie les modèles,
// lance l'auto-test si demandé et démarre la tâche IA. Ne bloque jamais ; en cas d'échec, l'IA est désactivée
// (message clair) et le firmware continue avec ses règles habituelles.
bool ia_setup(const IaOptions& options);
// AVANT Serial.begin() : agrandit le tampon d'émission série (le journal L ne bloque jamais la boucle).
void ia_serie_tx_taille();
// Après l'initialisation des capteurs : sans capteur de cuisse, l'IA est désactivée (« IA désactivée (cuisse absente) »).
void ia_capteur_cuisse(bool present);
// Remise à zéro de la session IA (dans calibrate()) : fusion (dont la référence de la dérive de fatigue),
// fenêtres, compteurs.
void ia_reset_session();
// À chaque passage de la boucle à 200 Hz, après le calcul de knee et de running.
// v = {shank.ax, shank.ay, shank.az, shank.gy, thigh.gy, knee}. Écrit dans `e50` les échantillons à 50 Hz
// produits (0, 1, rarement plus) et renvoie leur nombre : ce sont EXACTEMENT les valeurs données aux modèles
// (utilisées aussi par LOG_CSV). Ne bloque jamais.
int ia_mesure_200hz(uint32_t t_us, const float v[IA_N_CANAUX], bool course, bool calibration,
                    float e50[IA_MAX_ECHANTILLONS][IA_N_CANAUX]);
// Commandes à un caractère lues sur le port série : T (auto-test), B (banc de mesure), I (informations),
// P (plausibilité des capteurs), L (journal de données). Ne bloque jamais.
void ia_commandes();
// Champs ajoutés à la trame 1 Hz : ",A:<score>,Y:<classe>,Z:<état>,I:<ms>". Renvoie la longueur écrite.
int ia_champs_trame(char* buf, size_t taille);
// Dernier état de la fusion (0 OK, 1 ALERTE(type), 2 non classée) et type (1..5) : POINT D'ANCRAGE FUTUR,
// non utilisé pour l'instant (mode ombre).
uint8_t ia_etat_fusion(int8_t* type);
bool ia_active();

// Auto-test : rejoue les vecteurs de test (ia_autotest.cpp). Renvoie le nombre d'écarts (0 = OK, -1 = impossible).
int ia_autotest(bool afficher_details);
// Première fenêtre des vecteurs de test de l'autoencoder (mesures brutes), pour le banc de mesure.
const float (*ia_fenetre_test())[IA_N_CANAUX];
