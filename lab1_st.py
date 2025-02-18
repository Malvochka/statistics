import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, cauchy, poisson, uniform

# -------------------------------------------------------
# ЧАСТЬ 1. ГЕНЕРАЦИЯ ВЫБОРОК И ПОСТРОЕНИЕ ГИСТОГРАММ С PDF
# -------------------------------------------------------

# Фиксируем seed для воспроизводимости (по желанию)
np.random.seed(42)

# Параметры распределений
# Нормальное(0,1), Коши(0,1), Пуассон(lambda=10), Равномерное(-sqrt(3), sqrt(3))
lambda_pois = 10
a_unif, b_unif = -np.sqrt(3), np.sqrt(3)

# Размеры выборок, для которых строим гистограммы
sample_sizes_part1 = [10, 100, 1000]


def plot_distribution_samples(dist_name, sample_generator, pdf_or_pmf,
                              sample_sizes, is_discrete=False,
                              x_min=-5, x_max=5,
                              clip_for_cauchy=False):
    """
    dist_name: название распределения (строка)
    sample_generator: функция, генерирующая выборку нужного размера
    pdf_or_pmf: функция плотности (pdf) или функция pmf (для дискретных)
    sample_sizes: список размеров выборок
    is_discrete: True, если распределение дискретное (например, Пуассон)
    x_min, x_max: границы, в которых строим pdf (если не обрезаем)
    clip_for_cauchy: если True, то для Коши обрезаем хвосты по [1%, 99%]
                     чтобы гистограмма выглядела «презентабельно».
    """
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(15, 4))
    fig.suptitle(f"{dist_name}: Гистограммы и теоретическая плотность/pmf", fontsize=14)

    for i, n in enumerate(sample_sizes):
        ax = axes[i]
        # Генерируем выборку
        sample = sample_generator(n)

        # Если нужно обрезать хвосты для Коши (и именно для Коши)
        if clip_for_cauchy and ("Коши" in dist_name or "Cauchy" in dist_name):
            q1, q99 = np.percentile(sample, [1, 99])
            sample_clipped = sample[(sample >= q1) & (sample <= q99)]
            x_min_local, x_max_local = q1, q99
        else:
            sample_clipped = sample
            x_min_local, x_max_local = x_min, x_max

        if is_discrete:
            # Для дискретных распределений (например, Пуассон)
            min_val = int(np.min(sample_clipped))
            max_val = int(np.max(sample_clipped))
            bins = range(min_val, max_val + 2)
            # Гистограмма розового цвета
            ax.hist(sample_clipped, bins=bins, density=True, alpha=0.6,
                    color='pink', edgecolor='black')

            # pmf поверх гистограммы
            k_values = np.arange(min_val, max_val + 1)
            pmf_vals = [pdf_or_pmf(k) for k in k_values]
            ax.plot(k_values, pmf_vals, 'ro-', lw=2, label='Теоретическая pmf')
            ax.set_xlim([min_val - 0.5, max_val + 0.5])
        else:
            # Для непрерывных распределений
            ax.hist(sample_clipped, bins='auto', density=True, alpha=0.6,
                    color='pink', edgecolor='black')
            # pdf поверх гистограммы
            x_plot = np.linspace(x_min_local, x_max_local, 200)
            pdf_vals = pdf_or_pmf(x_plot)
            ax.plot(x_plot, pdf_vals, 'r-', lw=2, label='Теоретическая pdf')

        ax.set_title(f"n={n}")
        ax.legend()

    plt.tight_layout()
    plt.show()


# 1) Нормальное(0,1)
plot_distribution_samples(
    dist_name="Нормальное N(0,1)",
    sample_generator=lambda n: np.random.normal(0, 1, n),
    pdf_or_pmf=lambda x: norm.pdf(x, loc=0, scale=1),
    sample_sizes=sample_sizes_part1,
    is_discrete=False,
    x_min=-4, x_max=4,
    clip_for_cauchy=False
)

# 2) Коши(0,1)
plot_distribution_samples(
    dist_name="Коши C(0,1)",
    sample_generator=lambda n: np.random.standard_cauchy(n),
    pdf_or_pmf=lambda x: cauchy.pdf(x, loc=0, scale=1),
    sample_sizes=sample_sizes_part1,
    is_discrete=False,
    x_min=-10, x_max=10,
    clip_for_cauchy=True  # Для наглядности обрезаем хвосты
)

# 3) Пуассон(λ=10)
plot_distribution_samples(
    dist_name="Пуассон P(λ=10)",
    sample_generator=lambda n: np.random.poisson(lambda_pois, n),
    pdf_or_pmf=lambda k: poisson.pmf(k, mu=lambda_pois),
    sample_sizes=sample_sizes_part1,
    is_discrete=True,
    clip_for_cauchy=False
)

# 4) Равномерное(-√3, √3)
plot_distribution_samples(
    dist_name="Равномерное U(-√3, √3)",
    sample_generator=lambda n: np.random.uniform(a_unif, b_unif, n),
    pdf_or_pmf=lambda x: uniform.pdf(x, loc=a_unif, scale=(b_unif - a_unif)),
    sample_sizes=sample_sizes_part1,
    is_discrete=False,
    x_min=-3.5, x_max=3.5,
    clip_for_cauchy=False
)

# --------------------------------------------------------
# ЧАСТЬ 2. ВЫЧИСЛЕНИЕ СТАТИСТИК (СРЕДНЕЕ, МЕДИАНА, Z_R, Z_Q)
# ПОВТОРНОЕ ГЕНЕРИРОВАНИЕ (1000 РАЗ) И ОЦЕНКА СРЕДНИХ/ДИСПЕРСИЙ
# --------------------------------------------------------

sample_sizes_part2 = [20, 100, 1000]
num_experiments = 1000


def z_R(sample):
    """Полусумма крайних элементов: (min + max)/2."""
    return (np.min(sample) + np.max(sample)) / 2


def z_Q(sample):
    """Полусумма квартилей: (Q1 + Q3)/2."""
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    return (q1 + q3) / 2


def compute_statistics(sample):
    """Возвращает (mean, median, z_R, z_Q) для одной выборки."""
    return (
        np.mean(sample),
        np.median(sample),
        z_R(sample),
        z_Q(sample)
    )


def experiment(distribution_name, sample_generator, sizes, n_exp=1000):
    """
    Для заданного распределения и набора размеров выборки sizes
    генерирует выборки, считает статистики (mean, median, z_R, z_Q),
    повторяя n_exp раз, и возвращает средние и дисперсии по каждой статистике.
    """
    results = {}
    for n in sizes:
        stats_collection = []
        for _ in range(n_exp):
            s = sample_generator(n)
            stats_collection.append(compute_statistics(s))
        stats_array = np.array(stats_collection)
        mean_vals = np.mean(stats_array, axis=0)
        var_vals = np.var(stats_array, axis=0, ddof=1)  # несмещённая оценка
        results[n] = (mean_vals, var_vals)
    return distribution_name, results


def gen_normal(n):
    return np.random.normal(0, 1, n)


def gen_cauchy(n):
    return np.random.standard_cauchy(n)


def gen_poisson(n):
    return np.random.poisson(lambda_pois, n)


def gen_uniform(n):
    return np.random.uniform(a_unif, b_unif, n)


distributions = [
    ("Нормальное N(0,1)", gen_normal),
    ("Коши C(0,1)", gen_cauchy),
    (f"Пуассон(λ={lambda_pois})", gen_poisson),
    (f"Равномерное(-√3, √3)", gen_uniform)
]

all_results = []
for dist_name, dist_gen in distributions:
    name, res = experiment(dist_name, dist_gen, sample_sizes_part2, n_exp=num_experiments)
    all_results.append((name, res))

stat_names = ["Mean (x̄)", "Median", "z_R", "z_Q"]

for dist_name, dist_data in all_results:
    print(f"\n=== {dist_name} ===")
    print(" n | Statistic       |  Mean Value    |  Variance")
    print("------------------------------------------------------")
    for n in sample_sizes_part2:
        mean_vals, var_vals = dist_data[n]
        for i, st_name in enumerate(stat_names):
            print(f"{n:3d} | {st_name:<14} | {mean_vals[i]:>13.5f} | {var_vals[i]:>10.5f}")
    print("------------------------------------------------------")
