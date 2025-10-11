import numpy as np
import stumpy

# Синтетичен времеви ред с нарастваща верига
np.random.seed(0)
T = np.concatenate([
    np.sin(np.linspace(0, 2*np.pi, 200)),                # базова осцилация
    np.sin(np.linspace(0, 3*np.pi, 200)) + 0.5*np.arange(200)/200,  # дрейфящ фрагмент (верига)
    np.sin(np.linspace(0, 2*np.pi, 200))
])

m = 50  # дължина на подпоследователност (слайс)
mp = stumpy.stump(T, m)          # изчислява matrix profile (MP)
profile = mp[:, 0]               # първа колона: минималните разстояния

# Намиране на мотиви (удобна функция за начало)
motif_distances, motif_indices = stumpy.motifs(T, profile, max_motifs=3)

# Time Series Chains (примерно извикване — stumpy предлага utilities / tutorials)
chains = stumpy.time_series_chains(T, m)  # имена на функциите могат да варират в API

print("Намерени мотиви (индекси):", motif_indices)
print("Примерни връзки/вериги:", chains[:3])
