// ia_core.h — logique PURE de l'intégration IA (aucune dépendance à Arduino ni à TFLite Micro).
// Compilable et testée sur PC (edge-ai/tests_pc). Utilisée par ia_worksafe.cpp.
//
//  1. Mesure à 50 Hz : à partir des appels à 200 Hz de la boucle, produit exactement un échantillon de 6 canaux
//     par période de 20 ms (horloge en microsecondes), avec les mesures reçues PENDANT cette période (il est
//     produit au premier appel de la période suivante, donc 5 ms plus tard). Mode 1 : moyenne de ces mesures
//     (4 en général) ; mode 0 : la dernière (« une sur quatre », comme le LOG_CSV historique). Si aucune
//     mesure n'est arrivée pendant 20 ms (lecture du capteur ratée), la dernière valeur est répétée : on garde
//     ainsi 50 échantillons par seconde en moyenne.
//  2. Fenêtres : tampon circulaire de 50 échantillons x 6 canaux ; une fenêtre est prête dès 50 échantillons
//     de course, puis toutes les 25. Un échantillon n'entre dans le tampon que si « actif » (course en cours,
//     cuisse présente, pas de calibration) ; sinon le tampon est vidé.
//  3. Statistiques de plausibilité sur les 60 dernières secondes (60 paquets d'une seconde).
#pragma once
#include <stdint.h>

#define IA_N_CANAUX 6
#define IA_W 50
#define IA_PAS 25
#define IA_PERIODE_US 20000UL       // 50 Hz
#define IA_MAX_RATTRAPAGE 10        // au plus 10 échantillons répétés d'un coup (200 ms) ; au-delà : nouvelle origine

// ---- Mesure à 50 Hz ----
struct IaDecim {
  uint8_t mode;                     // 1 = moyenne, 0 = une mesure sur quatre (dernière reçue)
  bool demarre;
  uint32_t t_prochain;              // instant (µs) du prochain échantillon
  float somme[IA_N_CANAUX];
  float derniere[IA_N_CANAUX];      // valeur répétée si une période n'a reçu aucune mesure
  uint16_t n;                       // mesures accumulées depuis le dernier échantillon
  bool a_une_valeur;                // au moins une mesure reçue depuis le démarrage
  uint32_t repetes;                 // échantillons produits sans mesure nouvelle (compteur)
};
void ia_decim_init(IaDecim* d, uint8_t mode);
// Ajoute une mesure à 200 Hz (v = nullptr si la lecture a échoué). Écrit au plus `max_sorties` échantillons
// à 50 Hz dans `sorties` et renvoie leur nombre (souvent 0 ou 1).
int ia_decim_ajouter(IaDecim* d, uint32_t t_us, const float* v, float sorties[][IA_N_CANAUX], int max_sorties);

// ---- Fenêtres ----
struct IaFenetres {
  float tampon[IA_W][IA_N_CANAUX];  // circulaire
  uint16_t tete;                    // prochaine case à écrire
  uint16_t n;                       // échantillons dans le tampon (<= IA_W)
  uint16_t depuis;                  // échantillons ajoutés depuis la dernière fenêtre prête
  uint32_t fenetres;                // fenêtres prêtes depuis la remise à zéro
};
void ia_fenetres_vider(IaFenetres* f);
// Ajoute un échantillon si `actif`, sinon vide le tampon. Renvoie true si une fenêtre est prête.
bool ia_fenetres_ajouter(IaFenetres* f, const float e[IA_N_CANAUX], bool actif);
// Copie la fenêtre (les 50 derniers échantillons, du plus ancien au plus récent) : mesures BRUTES (g, °/s, °).
void ia_fenetres_extraire(const IaFenetres* f, float brut[IA_W][IA_N_CANAUX]);

// ---- Plausibilité (60 dernières secondes) ----
#define IA_PLAUS_SECONDES 60
struct IaPaquet {
  uint16_t n;
  uint16_t n_course;
  double somme[IA_N_CANAUX];
  double somme2[IA_N_CANAUX];
  float mini[IA_N_CANAUX];
  float maxi[IA_N_CANAUX];
};
struct IaPlausibilite {
  IaPaquet paquets[IA_PLAUS_SECONDES + 1];   // 60 secondes complètes + la seconde en cours
  uint8_t courant;
  uint8_t remplis;
  uint16_t dans_courant;            // échantillons dans le paquet courant (50 = une seconde)
};
struct IaStatsCanal {
  uint32_t n;
  float moyenne, ecart_type, mini, maxi;
  float ecart_moyenne_en_std;       // |moyenne - moyenne d'apprentissage| / écart-type d'apprentissage
  float rapport_std;                // écart-type / écart-type d'apprentissage
  bool suspect;
};
void ia_plaus_vider(IaPlausibilite* p);
void ia_plaus_ajouter(IaPlausibilite* p, const float e[IA_N_CANAUX], bool course);
// Statistiques sur les 60 dernières secondes, comparées aux constantes d'apprentissage (moyenne, écart-type).
// Règle : SUSPECT si la moyenne est à plus de 3 écarts-types d'apprentissage, ou si l'écart-type est hors de
// [0,3 ; 3] fois celui d'apprentissage. Renvoie le nombre d'échantillons (et ceux pris pendant la course).
uint32_t ia_plaus_calculer(const IaPlausibilite* p, const float moy_app[IA_N_CANAUX], const float std_app[IA_N_CANAUX],
                           IaStatsCanal stats[IA_N_CANAUX], uint32_t* n_course);
