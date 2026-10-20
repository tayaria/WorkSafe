// ia_core.cpp — logique PURE de l'intégration IA (voir ia_core.h). Aucune dépendance à Arduino ni à TFLite Micro.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")  // pas de multiplication-addition fusionnée (voir RAPPORT_INTEGRATION.md)
#endif
#include "ia_core.h"

#include <math.h>
#include <string.h>

// --------------------------------------------------------------------------
// Mesure à 50 Hz
// --------------------------------------------------------------------------
void ia_decim_init(IaDecim* d, uint8_t mode) {
  memset(d, 0, sizeof(*d));
  d->mode = mode ? 1 : 0;
}

int ia_decim_ajouter(IaDecim* d, uint32_t t_us, const float* v, float sorties[][IA_N_CANAUX], int max_sorties) {
  int k = 0;
  if (!d->demarre) {
    d->demarre = true;
    d->t_prochain = t_us + IA_PERIODE_US;
  } else {
    // trop en retard (boucle bloquée, calibration...) : on repart d'une nouvelle origine sans inventer de données
    if ((int32_t)(t_us - d->t_prochain) >= (int32_t)(IA_MAX_RATTRAPAGE * IA_PERIODE_US)) d->t_prochain = t_us;
    // chaque période de 20 ms écoulée donne un échantillon fait des mesures reçues PENDANT cette période
    while ((int32_t)(t_us - d->t_prochain) >= 0 && k < max_sorties) {
      if (d->n > 0) {
        for (int c = 0; c < IA_N_CANAUX; ++c) {
          sorties[k][c] = d->mode ? d->somme[c] / (float)d->n : d->derniere[c];
          d->derniere[c] = sorties[k][c];             // une éventuelle répétition reprend cette valeur
          d->somme[c] = 0.0f;
        }
        d->n = 0;
      } else if (d->a_une_valeur) {
        for (int c = 0; c < IA_N_CANAUX; ++c) sorties[k][c] = d->derniere[c];  // aucune mesure : on répète
        d->repetes++;
      } else {
        d->t_prochain += IA_PERIODE_US;               // rien à répéter au tout début
        continue;
      }
      d->t_prochain += IA_PERIODE_US;
      ++k;
    }
  }
  if (v != nullptr) {                                 // la mesure appartient à la période en cours
    for (int c = 0; c < IA_N_CANAUX; ++c) {
      d->somme[c] += v[c];
    }
    d->n++;
    d->a_une_valeur = true;
    if (!d->mode)
      for (int c = 0; c < IA_N_CANAUX; ++c) d->derniere[c] = v[c];
  }
  return k;
}

// --------------------------------------------------------------------------
// Fenêtres
// --------------------------------------------------------------------------
void ia_fenetres_vider(IaFenetres* f) {
  f->tete = 0;
  f->n = 0;
  f->depuis = 0;
}

bool ia_fenetres_ajouter(IaFenetres* f, const float e[IA_N_CANAUX], bool actif) {
  if (!actif) {
    if (f->n) ia_fenetres_vider(f);
    return false;
  }
  memcpy(f->tampon[f->tete], e, sizeof(float) * IA_N_CANAUX);
  f->tete = (uint16_t)((f->tete + 1) % IA_W);
  if (f->n < IA_W) f->n++;
  f->depuis++;
  if (f->n == IA_W && f->depuis >= IA_PAS) {
    f->depuis = 0;
    f->fenetres++;
    return true;
  }
  return false;
}

void ia_fenetres_extraire(const IaFenetres* f, float brut[IA_W][IA_N_CANAUX]) {
  for (int t = 0; t < IA_W; ++t)  // tete = plus ancien quand le tampon est plein
    memcpy(brut[t], f->tampon[(f->tete + t) % IA_W], sizeof(float) * IA_N_CANAUX);
}

// --------------------------------------------------------------------------
// Plausibilité
// --------------------------------------------------------------------------
static void paquet_vider(IaPaquet* q) {
  memset(q, 0, sizeof(*q));
  for (int c = 0; c < IA_N_CANAUX; ++c) {
    q->mini[c] = INFINITY;
    q->maxi[c] = -INFINITY;
  }
}

void ia_plaus_vider(IaPlausibilite* p) {
  for (int i = 0; i <= IA_PLAUS_SECONDES; ++i) paquet_vider(&p->paquets[i]);
  p->courant = 0;
  p->remplis = 0;
  p->dans_courant = 0;
}

void ia_plaus_ajouter(IaPlausibilite* p, const float e[IA_N_CANAUX], bool course) {
  IaPaquet* q = &p->paquets[p->courant];
  q->n++;
  if (course) q->n_course++;
  for (int c = 0; c < IA_N_CANAUX; ++c) {
    q->somme[c] += e[c];
    q->somme2[c] += (double)e[c] * (double)e[c];
    if (e[c] < q->mini[c]) q->mini[c] = e[c];
    if (e[c] > q->maxi[c]) q->maxi[c] = e[c];
  }
  if (++p->dans_courant >= 50) {  // une seconde complète : paquet suivant (le plus ancien est remplacé)
    p->dans_courant = 0;
    p->courant = (uint8_t)((p->courant + 1) % (IA_PLAUS_SECONDES + 1));
    paquet_vider(&p->paquets[p->courant]);
    if (p->remplis < IA_PLAUS_SECONDES) p->remplis++;
  }
}

uint32_t ia_plaus_calculer(const IaPlausibilite* p, const float moy_app[IA_N_CANAUX], const float std_app[IA_N_CANAUX],
                           IaStatsCanal stats[IA_N_CANAUX], uint32_t* n_course) {
  uint32_t n = 0, nc = 0;
  double s[IA_N_CANAUX] = {0}, s2[IA_N_CANAUX] = {0};
  float mi[IA_N_CANAUX], ma[IA_N_CANAUX];
  for (int c = 0; c < IA_N_CANAUX; ++c) {
    mi[c] = INFINITY;
    ma[c] = -INFINITY;
  }
  for (int i = 0; i <= IA_PLAUS_SECONDES; ++i) {
    const IaPaquet* q = &p->paquets[i];
    if (q->n == 0) continue;
    n += q->n;
    nc += q->n_course;
    for (int c = 0; c < IA_N_CANAUX; ++c) {
      s[c] += q->somme[c];
      s2[c] += q->somme2[c];
      if (q->mini[c] < mi[c]) mi[c] = q->mini[c];
      if (q->maxi[c] > ma[c]) ma[c] = q->maxi[c];
    }
  }
  for (int c = 0; c < IA_N_CANAUX; ++c) {
    IaStatsCanal* st = &stats[c];
    st->n = n;
    if (n == 0) {
      st->moyenne = st->ecart_type = st->mini = st->maxi = 0.0f;
      st->ecart_moyenne_en_std = st->rapport_std = 0.0f;
      st->suspect = false;
      continue;
    }
    const double m = s[c] / n;
    double v = s2[c] / n - m * m;
    if (v < 0) v = 0;
    st->moyenne = (float)m;
    st->ecart_type = (float)sqrt(v);
    st->mini = mi[c];
    st->maxi = ma[c];
    st->ecart_moyenne_en_std = (float)(fabs(m - moy_app[c]) / std_app[c]);
    st->rapport_std = st->ecart_type / std_app[c];
    st->suspect = st->ecart_moyenne_en_std > 3.0f || st->rapport_std < 0.3f || st->rapport_std > 3.0f;
  }
  if (n_course) *n_course = nc;
  return n;
}
