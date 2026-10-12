#!/usr/bin/env python3
"""
Générateur de dataset synthétique WorkSafe v2 (objectif 1 : détection d'anomalie).

Les 9 premières colonnes reproduisent EXACTEMENT le log CSV de l'ESP32 (50 Hz) :
    temps_ms, ax_tibia, ay_tibia, az_tibia, gy_tibia, gy_cuisse,
    angle_genou, temperature, niveau
Les colonnes suivantes sont la "vérité terrain" (absentes du vrai log) :
    athlete_id, session_id, stride_id, etiquette, anomalie, severite

Hypothèses sur les axes du MPU du tibia (à adapter à ton montage) :
    ax = axe long du tibia (≈ +1 g debout), ay = avant/arrière, az = latéral.

Usage :  python3 generer_dataset.py [dossier_sortie] [graine]
"""
import sys
import numpy as np
import pandas as pd

FS_SIM = 200          # fréquence de la boucle du firmware
DIV = 4               # LOG_CSV_DIV -> 50 Hz dans le CSV
KNEE_SIGN = 1

# Seuils du README (attention, danger)
S_FLEX = (120, 140)
S_HYPER = (-5, -10)
S_VEL = (700, 900)
S_IMPACT = (5, 8)
S_LAT = (300, 500)
S_TEMP = (2.5, 4.0)
S_FAT_CHOC = (0.25, 0.50)
S_FAT_AMP = (0.15, 0.25)
S_FAT_CAD = (0.10, 0.20)
BASE_STRIDES = 100

TYPES = ["instabilite", "faux_pas", "hyperextension", "boiterie"]


def gauss(phi, mu, sig):
    """Gaussienne périodique sur la phase [0,1)."""
    d = (phi - mu + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (d / sig) ** 2)


# --------------------------------------------------------------------------
# Athlètes virtuels
# --------------------------------------------------------------------------
def make_athlete(rng, aid):
    return dict(
        id=aid,
        cadence=rng.normal(86, 4.5),            # foulées/min de la jambe équipée
        k0=rng.normal(8, 2.5),                  # flexion en fin d'extension
        a1=rng.normal(24, 3.5),                 # flexion d'amortissement
        a2=np.clip(rng.normal(76, 6), 62, 90),                   # flexion de la phase aérienne
        sig2=rng.uniform(0.130, 0.150),
        thigh_amp=rng.normal(25, 3.5),          # oscillation de la cuisse (°)
        impact=np.clip(rng.normal(3.7, 0.4), 3.0, 4.3),   # pic d'impact (g)
        lat_coef=rng.uniform(0.05, 0.14),       # part latérale de l'impact
        tilt=rng.normal(0, 4),                  # désalignement du capteur (°)
        gyro_bias=rng.normal(0, 1.2, 2),
        t_rest=rng.normal(33.4, 0.6),
        dT_max=rng.uniform(0.9, 1.8),           # échauffement normal sur la séance
        cv_amp=rng.uniform(0.025, 0.045),       # variabilité naturelle (CV)
        cv_time=rng.uniform(0.012, 0.025),
        lat_gyro=rng.normal(70, 12),            # rotation latérale de base (°/s)
    )


# --------------------------------------------------------------------------
# Plans de séance : severité + type par foulée
# --------------------------------------------------------------------------
def plan_normal(n):
    return ["normal"] * n, np.zeros(n)


def plan_fatigue(n, rng, start_frac=0.25):
    """Normal au début puis fatigue qui s'installe PROGRESSIVEMENT."""
    s = np.zeros(n)
    i0 = int(n * start_frac)
    ramp = np.linspace(0, 1, n - i0) ** rng.uniform(1.0, 1.6)
    s[i0:] = ramp * rng.uniform(0.85, 1.0)
    return ["fatigue"] * n, s


def plan_episodes(n, rng, n_ep=None, strides_gap=(25, 55)):
    """Épisodes courts d'anomalies (instabilité, faux pas, ...) dans du normal."""
    types = ["normal"] * n
    sev = np.zeros(n)
    i = int(rng.integers(40, 70))
    while i < n - 15:
        t = TYPES[rng.integers(len(TYPES))]
        length = int(rng.integers(1, 3)) if t == "faux_pas" else int(rng.integers(10, 26))
        length = min(length, n - i)
        peak = rng.uniform(0.5, 1.0)
        if t == "faux_pas":
            prof = np.full(length, peak)
        else:
            up = max(2, length // 4)
            prof = np.concatenate([np.linspace(0.1, peak, up),
                                   np.full(length - up, peak)])[:length]
        for k in range(length):
            types[i + k] = t
            sev[i + k] = prof[k]
        i += length + int(rng.integers(*strides_gap))
    return types, sev


# --------------------------------------------------------------------------
# Génération d'une séance
# --------------------------------------------------------------------------
def simulate_session(ath, minutes, kind, rng, session_id, thermal_boost=1.0):
    n_strides = int(minutes * ath["cadence"] * 1.02) + 5
    if kind == "normal":
        types, sev = plan_normal(n_strides)
    elif kind == "fatigue":
        types, sev = plan_fatigue(n_strides, rng)
    elif kind == "episodes":
        types, sev = plan_episodes(n_strides, rng)
    else:
        raise ValueError(kind)

    total_samples_target = int(minutes * 60 * FS_SIM)
    drift_amp = 0.0
    drift_time = 0.0

    ax_l, ay_l, az_l, gy_l, gyc_l, kn_l = [], [], [], [], [], []
    stride_ids = []
    stride_feat = []   # par foulée : valeurs pour calculer le niveau
    n_samp = 0

    for si in range(n_strides):
        if n_samp >= total_samples_target:
            break
        s = sev[si]
        typ = types[si]
        fat = s if typ == "fatigue" else 0.0
        ins = s if typ == "instabilite" else 0.0
        stm = s if typ == "faux_pas" else 0.0
        hyp = s if typ == "hyperextension" else 0.0
        lim = s if typ == "boiterie" else 0.0

        # dérive lente naturelle (marche aléatoire bornée)
        drift_amp = 0.97 * drift_amp + rng.normal(0, 0.006)
        drift_time = 0.97 * drift_time + rng.normal(0, 0.004)

        # variabilité accrue avec la fatigue
        var_mult = 1 + 1.2 * fat + 0.6 * lim
        cad_scale = (1 - 0.17 * fat - 0.08 * lim)
        T = 60.0 / (ath["cadence"] * cad_scale)
        T *= 1 + drift_time + rng.normal(0, ath["cv_time"] * var_mult)
        if typ == "instabilite":
            T *= 1 + rng.normal(0, 0.03 * ins)
        if typ == "boiterie":
            T *= 1 + rng.normal(0, 0.05 * lim)
        if typ == "faux_pas":
            T *= 1 + rng.uniform(-0.2, 0.25) * stm
        T = max(T, 0.45)

        n = int(round(T * FS_SIM))
        t = np.arange(n) / FS_SIM
        phi = t / T

        amp_noise = 1 + drift_amp + rng.normal(0, ath["cv_amp"] * var_mult)
        a2 = ath["a2"] * amp_noise * (1 - 0.26 * fat - 0.25 * lim)
        a1 = ath["a1"] * amp_noise * (1 - 0.12 * fat - 0.40 * lim)
        sig2 = ath["sig2"]

        K = ath["k0"] + a1 * gauss(phi, 0.13, 0.06) + a2 * gauss(phi, 0.68, sig2)
        if typ == "hyperextension":
            K = K - hyp * 34.0 * gauss(phi, 0.97, 0.06) - hyp * 3.0
        if typ == "instabilite":
            K = K + rng.normal(0, 1.0) * ins * 6 * np.sin(2 * np.pi * (3.5 * phi + rng.uniform()))
        if typ == "faux_pas":
            K = K + stm * rng.uniform(8, 20) * gauss(phi, rng.uniform(0.05, 0.3), 0.05)
        dK = np.gradient(K, 1.0 / FS_SIM)          # vitesse sans bruit de mesure
        K = K + rng.normal(0, 0.4, n)              # bruit de mesure de l'angle

        th_amp = ath["thigh_amp"] * amp_noise * (1 - 0.10 * fat - 0.15 * lim)
        theta_th = th_amp * (np.sin(2 * np.pi * (phi - 0.10))
                             + 0.18 * np.sin(4 * np.pi * (phi - 0.05)))
        w_th = np.gradient(theta_th, 1.0 / FS_SIM)
        w_sh = w_th - dK                              # vitesse du tibia
        theta_sh = theta_th - K + ath["tilt"]         # inclinaison du tibia

        # Impact au contact du talon (début de foulée)
        P = ath["impact"] * (1 + 0.5 * fat + 0.40 * lim) \
            * (1 + rng.normal(0, 0.05 * var_mult))
        if typ == "faux_pas":
            P *= 1 + stm * rng.uniform(0.8, 1.6)
        if typ == "instabilite":
            P *= 1 + 0.25 * ins
        pulse = np.exp(-0.5 * ((t - 0.02) / 0.02) ** 2)
        ring = 0.30 * np.exp(-np.clip(t - 0.02, 0, None) / 0.05) \
            * np.sin(2 * np.pi * 17 * np.clip(t - 0.02, 0, None)) * (t > 0.02)
        imp = (P - 1.0) * pulse + (P - 1.0) * ring

        g = np.deg2rad(theta_sh)
        ax = np.cos(g) + imp + 0.25 * np.sin(2 * np.pi * (phi + 0.3)) * 0.5
        ay = np.sin(g) - 0.30 * imp + 0.35 * np.sin(2 * np.pi * (2 * phi + 0.1))
        lat_sign = 1 if rng.random() < 0.5 else -1
        az = (lat_sign * ath["lat_coef"] * (P - 1.0) * (pulse + ring)
              + 0.04 * np.sin(2 * np.pi * (phi + 0.2)))
        if typ == "instabilite":
            f_w = rng.uniform(3, 6)
            az = az + ins * 0.35 * np.sin(2 * np.pi * (f_w * phi + rng.uniform())) \
                + lat_sign * ins * 0.30 * (P - 1.0) * pulse
        if typ == "boiterie":
            az = az + lim * 0.06 * np.sin(2 * np.pi * (phi + 0.4))

        # rotation latérale (non loggée) pour le niveau
        lat_rate = ath["lat_gyro"] * (1 + 0.15 * fat) \
            + ins * rng.uniform(280, 420) + rng.normal(0, 12)

        ax_l.append(ax); ay_l.append(ay); az_l.append(az)
        gy_l.append(w_sh); gyc_l.append(w_th); kn_l.append(K)
        stride_ids.append(np.full(n, si))
        stride_feat.append(dict(
            stride=si, type=typ, sev=s, impact=float(np.max(ax)),
            kmax=float(K.max()), kmin=float(K.min()),
            vel=float(np.max(np.abs(dK))), cad=60.0 / T, lat=lat_rate,
            n=n, start=n_samp))
        n_samp += n

    cat = lambda L: np.concatenate(L)
    ax, ay, az = cat(ax_l), cat(ay_l), cat(az_l)
    gy, gyc, kn = cat(gy_l), cat(gyc_l), cat(kn_l)
    sid = cat(stride_ids)
    N = len(ax)

    # Bruit capteur non gaussien (Student t) + biais gyro résiduel
    ax += 0.035 * rng.standard_t(5, N)
    ay += 0.035 * rng.standard_t(5, N)
    az += 0.035 * rng.standard_t(5, N)
    gy += ath["gyro_bias"][0] + 3.5 * rng.standard_t(5, N)
    gyc += ath["gyro_bias"][1] + 3.5 * rng.standard_t(5, N)
    # Bruit de vibration accru pendant l'instabilité
    feat = pd.DataFrame(stride_feat)
    ins_mask = np.zeros(N)
    for _, r in feat.iterrows():
        if r["type"] == "instabilite":
            ins_mask[r["start"]: r["start"] + r["n"]] = r["sev"]
    gy += ins_mask * rng.normal(0, 25, N)
    az += ins_mask * rng.normal(0, 0.06, N)

    # Température : montée exponentielle, mesurée à 2 Hz (tenue entre deux lectures)
    time_s = np.arange(N) / FS_SIM
    sev_t = np.zeros(N)
    for _, r in feat.iterrows():
        if r["type"] == "fatigue":
            sev_t[r["start"]: r["start"] + r["n"]] = r["sev"]
    dT = ath["dT_max"] * (1 - np.exp(-time_s / (4 * 60)))
    dT = dT + thermal_boost * 1.6 * sev_t * (1 - np.exp(-time_s / (3 * 60)))
    temp_true = ath["t_rest"] + dT
    hold = (np.arange(N) // (FS_SIM // 2)) * (FS_SIM // 2)     # 2 Hz
    temp = temp_true[hold] + rng.normal(0, 0.09, N // (FS_SIM // 2) + 1)[hold // (FS_SIM // 2)]

    # ---- niveau (approximation des règles du firmware, sans hystérésis) ----
    f = feat.copy()
    base = f.iloc[:BASE_STRIDES]
    b_imp, b_amp, b_cad = base["impact"].mean(), base["kmax"].mean(), base["cad"].mean()
    roll = lambda c: f[c].rolling(10, min_periods=1).mean()
    r_imp, r_amp, r_cad = roll("impact"), roll("kmax"), roll("cad")
    f["d_imp"] = (r_imp - b_imp) / b_imp
    f["d_amp"] = (b_amp - r_amp) / b_amp
    f["d_cad"] = (b_cad - r_cad) / b_cad
    ready = f["stride"] >= BASE_STRIDES
    t_stride_end = (f["start"] + f["n"] - 1).clip(upper=N - 1)
    f["dtemp"] = temp_true[t_stride_end.values] - ath["t_rest"]

    def flags(row):
        att = dang = 0
        def chk(v, thr, hi=True):
            nonlocal att, dang
            if hi:
                if v >= thr[1]: dang += 1
                elif v >= thr[0]: att += 1
            else:
                if v <= thr[1]: dang += 1
                elif v <= thr[0]: att += 1
        chk(row["kmax"], S_FLEX)
        chk(row["kmin"], S_HYPER, hi=False)
        chk(row["vel"], S_VEL)
        chk(row["impact"], S_IMPACT)
        chk(row["lat"], S_LAT)
        chk(row["dtemp"], S_TEMP)
        if row["stride"] >= BASE_STRIDES:
            chk(row["d_imp"], S_FAT_CHOC)
            chk(row["d_amp"], S_FAT_AMP)
            chk(row["d_cad"], S_FAT_CAD)
        return 2 if (dang >= 1 or att >= 2) else (1 if att == 1 else 0)

    f["niveau"] = f.apply(flags, axis=1)

    niveau_s = np.zeros(N, dtype=int)
    etiq_s = np.empty(N, dtype=object)
    sev_s = np.zeros(N)
    for _, r in f.iterrows():
        sl = slice(r["start"], r["start"] + r["n"])
        niveau_s[sl] = r["niveau"]
        sev_s[sl] = r["sev"]
        etiq_s[sl] = r["type"] if r["sev"] >= 0.2 else "normal"

    # ---- décimation 200 Hz -> 50 Hz (comme LOG_CSV_DIV = 4) ----
    sel = np.arange(0, N, DIV)
    df = pd.DataFrame({
        "temps_ms": (np.arange(len(sel)) * (1000 // (FS_SIM // DIV))).astype(int),
        "ax_tibia": ax[sel].round(3),
        "ay_tibia": ay[sel].round(3),
        "az_tibia": az[sel].round(3),
        "gy_tibia": gy[sel].round(1),
        "gy_cuisse": gyc[sel].round(1),
        "angle_genou": (KNEE_SIGN * kn[sel]).round(1),
        "temperature": temp[sel].round(1),
        "niveau": niveau_s[sel],
        "athlete_id": ath["id"],
        "session_id": session_id,
        "stride_id": sid[sel],
        "etiquette": etiq_s[sel],
        "anomalie": (etiq_s[sel] != "normal").astype(int),
        "severite": sev_s[sel].round(2),
    })
    return df


# --------------------------------------------------------------------------
def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = np.random.default_rng(seed)

    n_train, n_val, n_test = 16, 4, 6
    athletes = [make_athlete(rng, i + 1) for i in range(n_train + n_val + n_test)]
    ath_train = athletes[:n_train]
    ath_val = athletes[n_train:n_train + n_val]
    ath_test = athletes[n_train + n_val:]

    sess = {"train": [], "val": [], "test": []}
    sid = 0

    def add(split, ath, minutes, kind):
        nonlocal sid
        sid += 1
        df = simulate_session(ath, minutes, kind, rng, sid)
        df.insert(0, "split", split)
        sess[split].append(df)

    for a in ath_train:                       # entraînement : NORMAL uniquement
        add("train", a, 3, "normal")
    for a in ath_val:                         # validation : seuil + réglages
        add("val", a, 2, "normal")
        add("val", a, 4, "episodes")
    for a in ath_test:                        # test : athlètes jamais vus
        add("test", a, 3, "normal")
        add("test", a, 8, "fatigue")
        add("test", a, 5, "episodes")

    for split, dfs in sess.items():
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(columns=["split"])
        path = f"{out}/dataset_{split}.csv"
        df.to_csv(path, index=False)
        print(path, df.shape, f"anomalies={df['anomalie'].mean():.1%}")


if __name__ == "__main__":
    main()
