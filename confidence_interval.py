# improved_ga_gam_with_uncertainty_penalty.py
# Full script integrating an uncertainty + sparsity penalty (Option A)
# Uses NSGA-II style multi-objective GA to optimize (RMSE, Penalty)
# Penalty = weighted combination of mean CI width (primary) and sparsity (secondary).
#
# Notes:
#  - This is a single-file integration based on your previous code.
#  - The uncertainty penalty uses model.partial_dependence(...) CIs when available.
#  - If partial_dependence fails for a term, that term contributes a high penalty (conservative).
#  - RMSE is normalized inside evaluator to stabilize scale across folds.

import random
from copy import deepcopy
import numpy as np
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM, s, l
from deap import base, creator, tools, algorithms
from deap.tools import sortNondominated
from baselines import test_baselines
from utils.dataset import RegressionDataset
from utils.visualization import visualize_models, visualize_pareto_front
from utils.repro import set_global_seed
import os
import json
from datetime import datetime, timezone
from typing import List, Optional

os.makedirs("outputs", exist_ok=True)


# ------------------------------
# 1. Chromosome & Builder
# ------------------------------
COMPONENT_TYPES = ["none", "linear", "spline"]


class ImprovedGAMChromosome:
    """
    Chromosome encoding: for each feature store a dict:
      { "type": "none"|"linear"|"spline", "knots": int|None, "lambda": float|None, "scale": bool }
    """
    def __init__(self, n_features, max_knots=20):
        self.genes = []
        self.n_features = n_features
        self.max_knots = max_knots
        self._initialize_smart()

    def _initialize_smart(self):
        """Bias initialization slightly towards linear/sparse solutions."""
        self.genes = []
        for _ in range(self.n_features):
            weights = [0.3, 0.5, 0.2]  # [none, linear, spline]
            component_type = random.choices(COMPONENT_TYPES, weights=weights)[0]
            gene = {
                "type": component_type,
                "knots": random.randint(8, self.max_knots) if component_type == "spline" else None,
                # sample lambda log-uniform for splines
                "lambda": (10 ** random.uniform(-2, 1)) if component_type == "spline" else None,
                "scale": random.choice([True, False])
            }
            self.genes.append(gene)

    def mutate(self, mutation_rate=0.15, generation=0, max_generations=100):
        adaptive_rate = mutation_rate * (1 - generation / max_generations)
        for gene in self.genes:
            if random.random() < adaptive_rate:
                if random.random() < 0.5:
                    # change type
                    gene["type"] = random.choice(COMPONENT_TYPES)
                    if gene["type"] == "spline":
                        gene["knots"] = random.randint(8, self.max_knots)
                        gene["lambda"] = 10 ** random.uniform(-2, 1)
                    else:
                        gene["knots"] = None
                        gene["lambda"] = None
                else:
                    # tweak hyperparams
                    if gene["type"] == "spline":
                        gene["knots"] = max(4, min(self.max_knots, gene.get("knots", 8) + random.randint(-2, 2)))
                        gene["lambda"] = max(1e-4, gene.get("lambda", 1.0) * random.uniform(0.7, 1.3))
                gene["scale"] = random.choice([True, False])

    def crossover(self, other, swap_prob=0.3):
        child1_genes, child2_genes = [], []
        for g1, g2 in zip(self.genes, other.genes):
            new_g1, new_g2 = {}, {}
            # type
            if random.random() < swap_prob:
                new_g1["type"], new_g2["type"] = g2["type"], g1["type"]
            else:
                new_g1["type"], new_g2["type"] = g1["type"], g2["type"]
            # scale
            if random.random() < swap_prob:
                new_g1["scale"], new_g2["scale"] = g2["scale"], g1["scale"]
            else:
                new_g1["scale"], new_g2["scale"] = g1["scale"], g2["scale"]
            # prefill
            new_g1["knots"], new_g1["lambda"] = None, None
            new_g2["knots"], new_g2["lambda"] = None, None
            # mix knots/lambda if both splines
            knots1, knots2 = g1.get("knots"), g2.get("knots")
            lam1, lam2 = g1.get("lambda"), g2.get("lambda")
            if new_g1["type"] == "spline" and new_g2["type"] == "spline":
                if random.random() < swap_prob:
                    new_g1["knots"], new_g2["knots"] = knots2, knots1
                else:
                    new_g1["knots"], new_g2["knots"] = knots1, knots2
                if random.random() < swap_prob:
                    new_g1["lambda"], new_g2["lambda"] = lam2, lam1
                else:
                    new_g1["lambda"], new_g2["lambda"] = lam1, lam2
            if new_g1["type"] == "spline" and new_g1["knots"] is None:
                new_g1["knots"], new_g1["lambda"] = random.randint(8, self.max_knots), 10 ** random.uniform(-2, 1)
            if new_g2["type"] == "spline" and new_g2["knots"] is None:
                new_g2["knots"], new_g2["lambda"] = random.randint(8, self.max_knots), 10 ** random.uniform(-2, 1)
            child1_genes.append(new_g1)
            child2_genes.append(new_g2)
        child1 = self.__class__(self.n_features, self.max_knots); child2 = self.__class__(self.n_features, self.max_knots)
        child1.genes = child1_genes; child2.genes = child2_genes
        return child1, child2


def is_valid_chromosome(chromosome, n_features, max_knots_per_feature, min_spline_knots=4):
    """Basic static checks to avoid impossible chromosomes."""
    if all(g["type"] == "none" for g in chromosome.genes):
        return False
    total_requested_knots = sum(g["knots"] for g in chromosome.genes if g["type"] == "spline" and g.get("knots"))
    if total_requested_knots > n_features * max_knots_per_feature:
        return False
    for g in chromosome.genes:
        if g["type"] == "spline" and (g.get("knots", 0) < min_spline_knots):
            return False
    return True


class ImprovedGAMBuilder:
    @staticmethod
    def build(chromosome, max_total_knots=100):
        """
        Construct a pygam LinearGAM from a chromosome.
        Returns None if builder couldn't allocate requested splines due to budget.
        """
        terms = None
        total_knots = 0
        for i, gene in enumerate(chromosome.genes):
            if gene["type"] == "none":
                continue
            elif gene["type"] == "linear":
                term = l(i)
            elif gene["type"] == "spline":
                available_knots = min(gene["knots"], max_total_knots - total_knots)
                if available_knots >= 4:
                    # use a conservative lam for building; actual smoothing recorded by model.lam
                    term = s(i, n_splines=available_knots, lam=float(gene.get("lambda", 0.6)))
                    total_knots += available_knots
                else:
                    return None
            if terms is None:
                terms = term
            else:
                terms += term
        if terms is None:
            # fallback to linear for all features
            terms = l(0)
            for i in range(1, chromosome.n_features):
                terms += l(i)
        gam = LinearGAM(terms, max_iter=100)
        return gam


# ------------------------------
# 2. Uncertainty + Sparsity Penalty (Option A)
# ------------------------------
def uncertainty_sparsity_penalty(model, X_ref, y_ref, n_features):
    """
    Primary components:
      - direct measure of uncertainty: mean CI width across active terms (CI width normalized by y_ref range)
      - sparsity: fraction of active features

    Returns:
      penalty in [0,1] where 1 = bad (high uncertainty and many active features).
    """
    import numpy as np

    if model is None:
        return 1.0

    # reference scale for normalization (use range of y_ref)
    y_range = float(np.ptp(y_ref)) if y_ref is not None and np.ptp(y_ref) > 0 else float(np.std(y_ref) + 1e-8)
    if y_range <= 0:
        y_range = 1.0

    ci_widths = []
    active_count = 0

    # iterate model.terms — note: term indexing matches the model's order
    term_index = 0
    for term in model.terms:
        # skip intercept-like terms (n_coefs == 1)
        try:
            n_coefs = getattr(term, "n_coefs", None)
            if n_coefs is None:
                # conservative: treat as active
                n_coefs = 2
        except Exception:
            n_coefs = 2

        # if term is effectively inactive (1 coef), skip CI calc
        if n_coefs <= 1:
            term_index += 1
            continue

        # attempt to compute partial dependence CI for this term
        try:
            # pygam.partial_dependence returns (grid, effect) or effect array and 'confidence' if width specified
            pd_result, conf = model.partial_dependence(term=term_index, width=0.95)
            # conf shape: (n_points, 2)
            # compute mean CI width for this term
            if conf is None:
                # fallback: if no conf returned, treat as high uncertainty
                mean_width = y_range  # maximal
            else:
                # conf could be shape (n_points, 2)
                conf = np.asarray(conf)
                widths = conf[:, 1] - conf[:, 0]
                mean_width = float(np.nanmean(np.abs(widths)))
        except Exception:
            # if PD fails, conservative: large uncertainty
            mean_width = y_range

        ci_widths.append(mean_width)
        active_count += 1
        term_index += 1

    # If model had no active (multi-coef) terms, penalize relative to emptiness
    if len(ci_widths) == 0:
        ci_score = 1.0  # maximum penalty (no supported shape)
    else:
        mean_ci = float(np.nanmean(ci_widths))
        # normalize by y range -> value in roughly [0, +inf), but we'll clip
        ci_score = mean_ci / (y_range + 1e-8)
        # clamp and squash into [0,1]
        ci_score = float(np.clip(ci_score, 0.0, 1.0))

    # sparsity: fraction of active features (active = term with n_coefs>1)
    # count number of model terms corresponding to active features:
    # approximate active features count by active_count (from above), but ensure <= n_features
    sparsity_frac = float(min(active_count, n_features)) / max(1, n_features)

    # combine: CI width primary (weight 0.7), sparsity secondary (0.3)
    penalty = 0.70 * ci_score + 0.30 * sparsity_frac
    penalty = float(np.clip(penalty, 0.0, 1.0))
    return penalty


# ------------------------------
# 3. Fitness Evaluator (multi-objective)
# ------------------------------
class ImprovedGAMFitnessEvaluator:
    @staticmethod
    def evaluate(chromosome, X_train, y_train, X_val, y_val,
                 n_repeats=2, cv_folds=5, max_knots_per_feature=20):
        """
        Returns (normalized_rmse, penalty) averaged across folds.
         - normalized_rmse = mean RMSE / rmse_scale  (rmse_scale = std(y_train+val))
         - penalty in [0,1] from uncertainty_sparsity_penalty (uses model PD CIs)
        """
        n_features = X_train.shape[1]
        if not is_valid_chromosome(chromosome, n_features, max_knots_per_feature):
            return 10.0, 1.0  # large RMSE, worst penalty

        # use combined train+val to get stable normalizer
        y_scale = float(np.std(np.hstack([y_train, y_val]))) + 1e-8
        if y_scale <= 0:
            y_scale = 1.0

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        n_samples = len(X_combined)
        fold_size = n_samples // cv_folds

        fold_rmse = []
        fold_pen = []

        for fold in range(cv_folds):
            v0 = fold * fold_size
            v1 = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
            X_val_fold = X_combined[v0:v1]
            X_train_fold = np.delete(X_combined, slice(v0, v1), axis=0)
            y_val_fold = y_combined[v0:v1]
            y_train_fold = np.delete(y_combined, slice(v0, v1))

            gam = ImprovedGAMBuilder.build(chromosome)
            if gam is None:
                fold_rmse.append(10.0)
                fold_pen.append(1.0)
                continue

            try:
                gam.fit(X_train_fold, y_train_fold)
                preds = gam.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
                norm_rmse = float(rmse) / y_scale

                # compute uncertainty + sparsity penalty using training fold as reference scale
                pen = uncertainty_sparsity_penalty(gam, X_train_fold, y_train_fold, n_features)

                fold_rmse.append(norm_rmse)
                fold_pen.append(pen)
            except Exception:
                fold_rmse.append(10.0)
                fold_pen.append(1.0)

        mean_norm_rmse = float(np.mean(fold_rmse))
        mean_penalty = float(np.mean(fold_pen))
        mean_norm_rmse = float(np.clip(mean_norm_rmse, 0.0, 100.0))
        mean_penalty = float(np.clip(mean_penalty, 0.0, 1.0))
        return mean_norm_rmse, mean_penalty


# ------------------------------
# 4. DEAP / NSGA-II Setup
# ------------------------------
def setup_improved_deap(n_features, X_train, y_train, X_val, y_val, population_size=80):
    # cleanup previous creator if exists
    if hasattr(creator, 'FitnessMin2'):
        del creator.FitnessMin2
    if hasattr(creator, 'Individual'):
        del creator.Individual

    # Multi-objective: minimize normalized RMSE, minimize penalty
    creator.create("FitnessMin2", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", ImprovedGAMChromosome, fitness=creator.FitnessMin2)

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, n_features=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return ImprovedGAMFitnessEvaluator.evaluate(ind, X_train, y_train, X_val, y_val)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", lambda ind1, ind2: ind1.crossover(ind2, swap_prob=0.3))

    def mutate_individual(individual, generation=0, max_generations=100):
        individual.mutate(mutation_rate=0.15, generation=generation, max_generations=max_generations)
        return individual,

    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


# ------------------------------
# 5. Knee / pickers
# ------------------------------
def pick_best_by_rmse(pop):
    # pop: list of Individuals (Pareto-front)
    return min(pop, key=lambda ind: ind.fitness.values[0])


def pick_best_by_penalty(pop):
    return min(pop, key=lambda ind: ind.fitness.values[1])


def pick_knee_solution(pop):
    """
    Pick knee solution by weighted distance to ideal point (0,0).
    Weights align with DEAP fitness weights: here we used (-1,-1) so equal weighting.
    """
    rmses = np.array([ind.fitness.values[0] for ind in pop])
    pens = np.array([ind.fitness.values[1] for ind in pop])

    # normalize both objectives to [0,1]
    r = (rmses - rmses.min()) / (rmses.max() - rmses.min() + 1e-8)
    p = (pens - pens.min()) / (pens.max() - pens.min() + 1e-8)

    # equal-weighted Euclidean distance to ideal (0,0)
    dist = np.sqrt(r * r + p * p)
    best_idx = np.argmin(dist)
    print("picked knee RMSE, penalty:", rmses[best_idx], pens[best_idx])
    return pop[best_idx]


# ------------------------------
# 6. GA run (Mu+Lambda variant)
# ------------------------------
def run_improved_ga(toolbox, population_size=80, n_generations=80):
    population = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: np.mean(fits, axis=0))
    stats.register("min", lambda fits: np.min(fits, axis=0))
    stats.register("max", lambda fits: np.max(fits, axis=0))

    hof = tools.HallOfFame(1)

    final_pop, logbook = algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=population_size,
        lambda_=population_size,
        cxpb=0.7,
        mutpb=0.2,
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    return final_pop, logbook, hof


# ------------------------------
# 7. Summarize / utility
# ------------------------------
def summarize_structure(chromosome, gam_model, feature_names):
    summary = {}
    if chromosome is not None:
        for i, gene in enumerate(chromosome.genes):
            summary[feature_names[i]] = gene["type"]
    if gam_model is not None:
        term_idx = 0
        for i, name in enumerate(feature_names):
            if term_idx >= len(gam_model.terms):
                summary[name + "_baseline"] = "none"
            else:
                term = gam_model.terms[term_idx]
                lam_value = np.mean(term.lam) if isinstance(term.lam, (list, np.ndarray)) else term.lam
                if term.n_coefs == 1:
                    summary[name + "_baseline"] = "none"
                elif lam_value is not None and lam_value > 1e4:
                    summary[name + "_baseline"] = "linear"
                else:
                    summary[name + "_baseline"] = "spline"
                term_idx += 1
    return summary


def build_and_fit_safe(chrom, X_combined, y_combined):
    gam = ImprovedGAMBuilder.build(chrom)
    if gam is None:
        return None
    try:
        gam.fit(X_combined, y_combined)
        return gam
    except Exception:
        return None


# ------------------------------
# 8. Main runner
# ------------------------------
def improved_main(seeds: Optional[List[int]] = None, population_size: int = 80, n_generations: int = 80):
    env_seeds = os.environ.get("SEEDS")
    if env_seeds:
        seeds_list = [int(s.strip()) for s in env_seeds.split(",") if s.strip()]
    elif seeds is not None:
        seeds_list = list(seeds)
    else:
        seeds_list = [42]

    aggregate_rows = []
    aggregate_path = "outputs/improved_results_all_seeds.md"
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

    for SEED in seeds_list:
        print(f"\n=== Running improved GA for seed {SEED} ===")
        set_global_seed(SEED)

        dataset = RegressionDataset(default="california", random_state=SEED, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()
        n_features = X_train.shape[1]

        toolbox = setup_improved_deap(n_features, X_train, y_train, X_val, y_val, population_size)
        final_pop, logbook, hof = run_improved_ga(toolbox, population_size, n_generations)

        # Pareto front (first front)
        pareto_front = sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
        visualize_pareto_front(pareto_front, f'outputs/improved_pareto_front_seed{SEED}.png')

        best_by_rmse = pick_best_by_rmse(pareto_front)
        best_by_penalty = pick_best_by_penalty(pareto_front)
        knee = pick_knee_solution(pareto_front)

        best_chrom_knee = deepcopy(knee)
        best_chrom_rmse = deepcopy(best_by_rmse)
        best_chrom_penalty = deepcopy(best_by_penalty)

        best_chrom_knee = greedy_prune(best_chrom_knee, X_train, y_train, X_val, y_val, tol=1e-3)
        best_chrom_rmse = greedy_prune(best_chrom_rmse, X_train, y_train, X_val, y_val, tol=1e-3)
        best_chrom_penalty = greedy_prune(best_chrom_penalty, X_train, y_train, X_val, y_val, tol=1e-3)

        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])

        best_gam_knee = build_and_fit_safe(best_chrom_knee, X_combined, y_combined)
        best_gam_rmse = build_and_fit_safe(best_chrom_rmse, X_combined, y_combined)
        best_gam_penalty = build_and_fit_safe(best_chrom_penalty, X_combined, y_combined)

        def eval_on_test(gam):
            if gam is None:
                return float('nan')
            preds = gam.predict(X_test)
            return float(np.sqrt(mean_squared_error(y_test, preds)))

        rmse_knee = eval_on_test(best_gam_knee)
        rmse_best_rmse = eval_on_test(best_gam_rmse)
        rmse_best_penalty = eval_on_test(best_gam_penalty)

        print(f"Improved GA-GAM (knee) Test RMSE: {rmse_knee:.4f}")
        print(f"Improved GA-GAM (best_by_rmse) Test RMSE: {rmse_best_rmse:.4f}")
        print(f"Improved GA-GAM (best_by_penalty) Test RMSE: {rmse_best_penalty:.4f}")

        # Baselines
        decision_tree_model, pygam_model, rmse_dt, rmse_gam = test_baselines(X_train, X_val, X_test, y_train, y_val, y_test, seed=SEED)

        penalty_baseline = uncertainty_sparsity_penalty(pygam_model.model, X_train, y_train, n_features)
        penalty_knee = uncertainty_sparsity_penalty(best_gam_knee, X_train, y_train, n_features)
        penalty_rmse = uncertainty_sparsity_penalty(best_gam_rmse, X_train, y_train, n_features)
        penalty_penalty = uncertainty_sparsity_penalty(best_gam_penalty, X_train, y_train, n_features)

        print(f"Penalty (baseline): {penalty_baseline:.4f}")
        print(f"Penalty (knee): {penalty_knee:.4f}")
        print(f"Penalty (best_by_rmse): {penalty_rmse:.4f}")
        print(f"Penalty (best_by_penalty): {penalty_penalty:.4f}")

        # Visualize feature effects
        models = [best_gam_knee, best_gam_rmse, best_gam_penalty, decision_tree_model, pygam_model]
        model_names = ["GA (knee)", "GA (best_by_rmse)", "GA (best_by_penalty)", "Decision Tree", "Baseline PyGAM"]
        chromosomes = [best_chrom_knee, best_chrom_rmse, best_chrom_penalty, None, None]
        vis_path = f'outputs/improved_feature_effects_seed{SEED}'
        visualize_models(X_test, feature_names, models, model_names, chromosomes, vis_path)

        # Summaries
        summary_knee = summarize_structure(best_chrom_knee, None, feature_names)
        summary_rmse = summarize_structure(best_chrom_rmse, None, feature_names)
        summary_penalty = summarize_structure(best_chrom_penalty, None, feature_names)
        summary_baseline = summarize_structure(None, pygam_model.model, feature_names)

        # Save markdown results
        results_path = f"outputs/improved_results_seed{SEED}.md"
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        md_lines = []
        md_lines.append(f"# Improved Run results — seed {SEED}\n")
        md_lines.append(f"- Date (UTC): {now}\n")
        md_lines.append(f"- Improved GA-GAM (knee) Final Test RMSE: {rmse_knee:.4f}\n")
        md_lines.append(f"- Improved GA-GAM (best_by_rmse) Final Test RMSE: {rmse_best_rmse:.4f}\n")
        md_lines.append(f"- Improved GA-GAM (best_by_penalty) Final Test RMSE: {rmse_best_penalty:.4f}\n")
        md_lines.append(f"- Decision Tree Test RMSE: {rmse_dt:.4f}\n")
        md_lines.append(f"- Baseline PyGAM Test RMSE: {rmse_gam:.4f}\n")
        md_lines.append(f"- Penalty (baseline): {penalty_baseline:.4f}\n")
        md_lines.append(f"- Penalty (knee): {penalty_knee:.4f}\n")
        md_lines.append(f"- Penalty (best_by_rmse): {penalty_rmse:.4f}\n")
        md_lines.append(f"- Penalty (best_by_penalty): {penalty_penalty:.4f}\n")

        md_lines.append("\n## Generation Log\n")
        md_lines.append("| Gen | Best RMSE | Average RMSE |\n|---:|---:|---:|\n")
        for entry in logbook:
            best_rmse = float(np.asarray(entry.get('min'))[0]) if entry.get('min') is not None else float('nan')
            avg_rmse = float(np.asarray(entry.get('avg'))[0]) if entry.get('avg') is not None else float('nan')
            md_lines.append(f"| {entry['gen']} | {best_rmse:.6f} | {avg_rmse:.6f} |\n")

        md_lines.append("\n## Model Structure Summaries\n")
        md_lines.append("### GA (knee)\n")
        for feature, ftype in summary_knee.items():
            md_lines.append(f"- **{feature}**: {ftype}\n")
        md_lines.append("\n### GA (best_by_rmse)\n")
        for feature, ftype in summary_rmse.items():
            md_lines.append(f"- **{feature}**: {ftype}\n")
        md_lines.append("\n### GA (best_by_penalty)\n")
        for feature, ftype in summary_penalty.items():
            md_lines.append(f"- **{feature}**: {ftype}\n")
        md_lines.append("\n### Baseline PyGAM\n")
        for feature, ftype in summary_baseline.items():
            md_lines.append(f"- **{feature}**: {ftype}\n")

        md_lines.append("\n## Best Chromosomes (JSON)\n")
        def safe_json(gen):
            try:
                return json.dumps(gen, indent=2)
            except Exception:
                return str(gen)
        md_lines.append("### GA (knee)\n```json\n")
        md_lines.append(safe_json(best_chrom_knee.genes))
        md_lines.append("\n```\n")
        md_lines.append("### GA (best_by_rmse)\n```json\n")
        md_lines.append(safe_json(best_chrom_rmse.genes))
        md_lines.append("\n```\n")
        md_lines.append("### GA (best_by_penalty)\n```json\n")
        md_lines.append(safe_json(best_chrom_penalty.genes))
        md_lines.append("\n```\n")

        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"Saved results to {results_path}")

        aggregate_rows.append({
            "seed": SEED,
            "rmse_knee": float(rmse_knee) if not np.isnan(rmse_knee) else float('nan'),
            "rmse_best_by_rmse": float(rmse_best_rmse) if not np.isnan(rmse_best_rmse) else float('nan'),
            "rmse_best_by_penalty": float(rmse_best_penalty) if not np.isnan(rmse_best_penalty) else float('nan'),
            "rmse_dt": float(rmse_dt),
            "rmse_gam": float(rmse_gam),
            "improvement_over_baseline": float(rmse_gam - rmse_knee) if not np.isnan(rmse_knee) else float('nan'),
            "vis": vis_path,
            "results_md": results_path,
        })

    # aggregate summary
    agg_lines = []
    agg_lines.append("# Improved Aggregate results for seeds\n")
    agg_lines.append("| Seed | GA (knee) RMSE | GA (best_by_rmse) RMSE | GA (best_by_penalty) RMSE | DecisionTree RMSE | PyGAM RMSE | Improvement |\n")
    agg_lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in aggregate_rows:
        improvement = r.get('improvement_over_baseline', float('nan'))
        improvement_text = f"{improvement:+.4f}"
        if not np.isnan(improvement) and improvement > 0:
            improvement_text += " Success:"
        else:
            improvement_text += " Failure:"
        agg_lines.append(f"| {r['seed']} | {r.get('rmse_knee', float('nan')):.4f} | {r.get('rmse_best_by_rmse', float('nan')):.4f} | {r.get('rmse_best_by_penalty', float('nan')):.4f} | {r['rmse_dt']:.4f} | {r['rmse_gam']:.4f} | {improvement_text} |\n")

    avg_improvement = float(np.nanmean([r.get('improvement_over_baseline', float('nan')) for r in aggregate_rows]))
    agg_lines.append(f"\n**Average improvement over baseline PyGAM: {avg_improvement:+.4f}**\n")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        f.write("\n".join(agg_lines))
    print(f"Saved aggregate results to {aggregate_path}")
    return aggregate_rows

def greedy_prune(chromosome, X_train, y_train, X_val, y_val, tol=1e-3):
    best = deepcopy(chromosome)
    gam = ImprovedGAMBuilder.build(best)
    if gam is None:
        return best
    gam.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    best_rmse = np.sqrt(mean_squared_error(y_val, gam.predict(X_val)))
    for i, g in enumerate(best.genes):
        if g["type"] == "spline":
            old = deepcopy(g)
            best.genes[i]["type"] = "none"
            candidate = ImprovedGAMBuilder.build(best)
            if candidate is None:
                best.genes[i] = old
                continue
            try:
                candidate.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
                rmse = np.sqrt(mean_squared_error(y_val, candidate.predict(X_val)))
                if rmse <= best_rmse + tol:
                    best_rmse = rmse
                    # keep removal
                else:
                    best.genes[i] = old
            except Exception:
                best.genes[i] = old
    return best



if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING IMPROVED GA-GAM (UNCERTAINTY + SPARSITY PENALTY)")
    print("=" * 60)
    #improved_results = improved_main(seeds=[42, 7, 123, 225, 729], population_size=80, n_generations=20)
    improved_results = improved_main(seeds=[729], population_size=80, n_generations=20)
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for result in improved_results:
        seed = result['seed']
        improvement = result['improvement_over_baseline']
        status = "Success" if improvement > 0 else "Failure"
        print(f"Seed {seed}: {status} (Improvement: {improvement:+.4f})")

    avg_improvement = np.nanmean([r.get('improvement_over_baseline', float('nan')) for r in improved_results])
    print(f"\nOverall average improvement: {avg_improvement:+.4f}")
    if avg_improvement > 0:
        print("SUCCESS: Improved GA-GAM outperforms baseline PyGAM on average!")
    else:
        print("No consistent improvement over baseline PyGAM.")

