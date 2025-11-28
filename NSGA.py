# ------------------------------
# Imports
# ------------------------------
import random
from xml.parsers.expat import model
import numpy as np
from copy import deepcopy
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 2. Improved GAM Chromosome
# ------------------------------
COMPONENT_TYPES = ["none", "linear", "spline"]

class ImprovedGAMChromosome:
    """
    Enhanced chromosome encoding for univariate regression GAM.
    """
    def __init__(self, n_features, max_knots=20):
        self.genes = []
        self.n_features = n_features
        self.max_knots = max_knots
        self._initialize_smart()

    def _initialize_smart(self):
        """Smarter initialization biased towards useful configurations"""
        self.genes = []
        for _ in range(self.n_features):
            # Bias initialization: prefer splines, then linear, then none
            weights = [0.3, 0.5, 0.2]  # [none, linear, spline]
            component_type = random.choices(COMPONENT_TYPES, weights=weights)[0]
            
            gene = {
                "type": component_type,
                "knots": random.randint(8, self.max_knots) if component_type == "spline" else None,
                "lambda": (10 ** random.uniform(-2, 1)) if component_type == "spline" else None,
                "scale": random.choice([True, False])
            }
            self.genes.append(gene)

    def mutate(self, mutation_rate=0.15, generation=0, max_generations=100):
        """Adaptive mutation that decreases over time"""
        adaptive_rate = mutation_rate * (1 - generation / max_generations)
        
        for gene in self.genes:
            if random.random() < adaptive_rate:
                if random.random() < 0.5:  # 50% chance to change type
                    gene["type"] = random.choice(COMPONENT_TYPES[1:2])
                    if gene["type"] == "spline":
                        gene["knots"] = random.randint(8, self.max_knots)
                        gene["lambda"] = (10 ** random.uniform(-2, 1)) 
                    else:
                        gene["knots"] = None
                        gene["lambda"] = None
                else:  # 30% chance to tweak hyperparameters
                    if gene["type"] == "spline":
                        # Small perturbations
                        gene["knots"] = max(4, min(self.max_knots, 
                                                 gene["knots"] + random.randint(-2, 2)))
                        gene["lambda"] = max(0.01, gene["lambda"] * random.uniform(0.7, 1.3))
                gene["scale"] = random.choice([True, False])

    def crossover(self, other, swap_prob=0.3):
        """
        Uniform crossover with lower swap probability for more conservative mixing.
        """
        child1_genes = []
        child2_genes = []

        for g1, g2 in zip(self.genes, other.genes):
            new_g1 = {}
            new_g2 = {}

            # --- 1. Crossover type ---
            if random.random() < swap_prob:
                new_g1["type"] = g2["type"]
                new_g2["type"] = g1["type"]
            else:
                new_g1["type"] = g1["type"]
                new_g2["type"] = g2["type"]

            # --- 2. Crossover scale (boolean) ---
            if random.random() < swap_prob:
                new_g1["scale"] = g2["scale"]
                new_g2["scale"] = g1["scale"]
            else:
                new_g1["scale"] = g1["scale"]
                new_g2["scale"] = g2["scale"]

            # --- 3. Crossover knots and lambda only if spline ---
            for new_gene in (new_g1, new_g2):
                # Pre-fill so keys always exist
                new_gene["knots"] = None
                new_gene["lambda"] = None

            # Parent parameter values
            knots1, knots2 = g1.get("knots"), g2.get("knots")
            lam1, lam2 = g1.get("lambda"), g2.get("lambda")

            # If both parents use spline → mix their hyperparameters
            if new_g1["type"] == "spline" and new_g2["type"] == "spline":
                # --- Mix knots ---
                if random.random() < swap_prob:
                    new_g1["knots"] = knots2
                    new_g2["knots"] = knots1
                else:
                    new_g1["knots"] = knots1
                    new_g2["knots"] = knots2

                # --- Mix lambda ---
                if random.random() < swap_prob:
                    new_g1["lambda"] = lam2
                    new_g2["lambda"] = lam1
                else:
                    new_g1["lambda"] = lam1
                    new_g2["lambda"] = lam2

            # If one or both children became spline while parent wasn't spline
            # reinitialize parameters safely
            if new_g1["type"] == "spline" and new_g1["knots"] is None:
                new_g1["knots"] = random.randint(8, self.max_knots)
                new_g1["lambda"] = (10 ** random.uniform(-2, 1)) 

            if new_g2["type"] == "spline" and new_g2["knots"] is None:
                new_g2["knots"] = random.randint(8, self.max_knots)
                new_g2["lambda"] = (10 ** random.uniform(-2, 1)) 
            # Append to the child gene lists
            child1_genes.append(new_g1)
            child2_genes.append(new_g2)

        # Create child chromosomes using the same class
        child1 = self.__class__(self.n_features, self.max_knots)
        child2 = self.__class__(self.n_features, self.max_knots)
        child1.genes = child1_genes
        child2.genes = child2_genes
        return child1, child2


def is_valid_chromosome(chromosome, n_features, max_knots_per_feature, min_spline_knots=4):
    # Require at least one non-none term (or you could allow but penalize)
    if all(g["type"] == "none" for g in chromosome.genes):
        return False

    # Check total requested knots not exceeding what's sensibly allowed
    total_requested_knots = 0
    for g in chromosome.genes:
        if g["type"] == "spline" and g.get("knots") is not None:
            total_requested_knots += g["knots"]
    max_total = n_features * max_knots_per_feature
    if total_requested_knots > max_total:
        return False

    # Ensure any spline gene has at least `min_spline_knots`
    for g in chromosome.genes:
        if g["type"] == "spline":
            if g.get("knots", 0) < min_spline_knots:
                return False

    return True

class ImprovedGAMBuilder:
    @staticmethod
    def build(chromosome, max_total_knots=100):
        """Build GAM with constraints on total complexity"""
        terms = None
        total_knots = 0
        
        for i, gene in enumerate(chromosome.genes):
            if gene["type"] == "none":
                continue
                
            elif gene["type"] == "linear":
                term = l(i)
                
            elif gene["type"] == "spline":
                # Constrain total knots to prevent overfitting
                available_knots = min(gene["knots"], max_total_knots - total_knots)
                if available_knots >= 4:  # Minimum sensible knots
                    term = s(i, n_splines=available_knots, lam=0.6)
                    total_knots += available_knots
                else:
                    # Not enough budget for this spline: mark builder invalid
                    return None

                    
            if terms is None:
                terms = term
            else:
                terms += term
                
        if terms is None:
            # Fallback: linear for all (we allow this), but not if it was supposed to be a spline-heavy chromosome:
            terms = l(0)
            for i in range(1, chromosome.n_features):
                terms += l(i)


        gam = LinearGAM(terms, max_iter=100)
        return gam


# ------------------------------
# 4. Improved Fitness Evaluator
# ------------------------------
class ImprovedGAMFitnessEvaluator:
    @staticmethod
    def evaluate(chromosome, X_train, y_train, X_val, y_val, 
                 n_repeats=2, cv_folds=5, max_knots_per_feature=20):
        """
        Return (normalized_rmse, normalized_penalty) averaged across folds.
        Normalized RMSE = rmse / rmse_scale where rmse_scale = std(y_train)
        Normalized penalty in [0,1].
        """
        # quick pre-validation
        n_features = X_train.shape[1]
        if not is_valid_chromosome(chromosome, n_features, max_knots_per_feature):
            return 10.0, 1.0  # bad normalized RMSE, worst penalty

        # RMSE scale to normalize (avoid per-fold small numbers)
        rmse_scale = np.std(np.hstack([y_train, y_val])) + 1e-8
        if rmse_scale == 0:
            rmse_scale = 1.0

        fold_results = []
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        n_samples = len(X_combined)
        fold_size = n_samples // cv_folds

        for fold in range(cv_folds):
            v0 = fold * fold_size
            v1 = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
            X_val_fold = X_combined[v0:v1]
            X_train_fold = np.delete(X_combined, slice(v0, v1), axis=0)
            y_val_fold = y_combined[v0:v1]
            y_train_fold = np.delete(y_combined, slice(v0, v1))

            # Build GAM and check builder validity
            gam = ImprovedGAMBuilder.build(chromosome)
            if gam is None:
                # invalid (builder couldn't allocate knots)
                fold_results.append((10.0, 1.0))
                continue

            try:
                gam.fit(X_train_fold, y_train_fold)
                preds = gam.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
                norm_rmse = float(rmse) / rmse_scale

                penalty = complexity_penalty(gam, n_features=n_features, max_knots_per_feature=max_knots_per_feature)

                fold_results.append((norm_rmse, penalty))
            except Exception as e:
                # fitting failed, give large bad normalized RMSE and worst penalty
                fold_results.append((10.0, 1.0))

        mean_rmse = float(np.mean([f[0] for f in fold_results]))
        mean_penalty = float(np.mean([f[1] for f in fold_results]))

        # Additional stabilization: clip to reasonable ranges
        mean_rmse = float(np.clip(mean_rmse, 0.0, 100.0))
        mean_penalty = float(np.clip(mean_penalty, 0.0, 1.0))
        return mean_rmse, mean_penalty


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

# ------------------------------
# 5. Improved GA Setup with DEAP
# ------------------------------
def setup_improved_deap(n_features, X_train, y_train, X_val, y_val, population_size=80):
    # Clean up existing classes to avoid conflicts
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual

    # NSGA-II Fitness: Minimize RMSE, Minimize Complexity
    creator.create("FitnessMin2", base.Fitness, weights=(-5.0, -1.0))
    creator.create("Individual", ImprovedGAMChromosome, fitness=creator.FitnessMin2)


    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, n_features=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        rmse, penalty = ImprovedGAMFitnessEvaluator.evaluate(ind, 
                                                            X_train, y_train, 
                                                            X_val, y_val)
        return (rmse, penalty)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", lambda ind1, ind2: ind1.crossover(ind2, swap_prob=0.3))
    
    def mutate_individual(individual, generation=0, max_generations=100):
        individual.mutate(mutation_rate=0.15, generation=generation, max_generations=max_generations)
        return individual,
    
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selNSGA2)

    return toolbox

def pick_best_by_rmse(pop):
    return min(pop, key=lambda ind: ind.fitness.values[0])

def pick_best_by_penalty(pop):
    return min(pop, key=lambda ind: ind.fitness.values[1])

"""
def pick_knee_solution(pop):
    # Normalize rmse and penalty
    rmses = np.array([ind.fitness.values[0] for ind in pop])
    pens  = np.array([ind.fitness.values[1] for ind in pop])

    r = (rmses - rmses.min()) / (rmses.max() - rmses.min() + 1e-8)
    p = (pens  - pens.min())  / (pens.max()  - pens.min()  + 1e-8)

    # Distance to ideal point (0,0)
    dist = np.sqrt(3*3*r*r + p*p)
    best_idx = np.argmin(dist)
    print(rmses[best_idx], pens[best_idx])  # restore original fitness
    return pop[best_idx]
"""

def pick_knee_solution(pop):
    # Extract objective values
    rmses = np.array([ind.fitness.values[0] for ind in pop])
    pens  = np.array([ind.fitness.values[1] for ind in pop])

    # Normalize both objectives independently
    r = (rmses - rmses.min()) / (rmses.max() - rmses.min() + 1e-8)
    p = (pens  - pens.min())  / (pens.max()  - pens.min()  + 1e-8)

    # Weights taken from DEAP fitness weighting (-3, -1)
    w_rmse = 3.0
    w_pen  = 1.0

    # Weighted distance to ideal point (0,0)
    dist = np.sqrt((w_rmse * r)**2 + (w_pen * p)**2)

    # Pick individual closest to weighted ideal point
    best_idx = np.argmin(dist)

    print("knee rmse:", rmses[best_idx], "penalty:", pens[best_idx])
    return pop[best_idx]

def run_improved_ga(toolbox, population_size=80, n_generations=80):
    """Run improved GA with better strategy"""
    population = toolbox.population(n=population_size)
    
    # Statistics for tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: np.mean(fits, axis=0))
    stats.register("min", lambda fits: np.min(fits, axis=0))
    stats.register("max", lambda fits: np.max(fits, axis=0))

    
    # Hall of fame to keep best individuals
    hof = tools.HallOfFame(1)
    
    # Use eaSimple with custom parameters
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

def summarize_structure(chromosome, gam_model, feature_names):
    """
    Return a dict: feature -> type ('none', 'linear', 'spline')
    """
    summary = {}
    # GA-GAM
    if chromosome is not None:
        for i, gene in enumerate(chromosome.genes):
            summary[feature_names[i]] = gene["type"]
    
    # Baseline GAM
    if gam_model is not None:
        term_idx = 0
        for i, name in enumerate(feature_names):
            if term_idx >= len(gam_model.terms):
                summary[name + "_baseline"] = "none"
            else:
                term = gam_model.terms[term_idx]
                # Get a scalar lambda
                lam_value = np.mean(term.lam) if isinstance(term.lam, (list, np.ndarray)) else term.lam
                if term.n_coefs == 1:
                    summary[name + "_baseline"] = "none"
                elif lam_value > 1e4:
                    summary[name + "_baseline"] = "linear"
                else:
                    summary[name + "_baseline"] = "spline"
                term_idx += 1
    return summary

def complexity_penalty(model, n_features, max_knots_per_feature):
    """
    Revised Strong Sparsity Penalty focusing on:
      1. Minimizing the total number of knots (highest priority).
      2. Minimizing the number of active features.
      3. Maximizing lambda (smoothing).
      4. Minimizing edof (wiggliness).
    """

    import numpy as np

    if model is None:
        return 1.0

    total_knots = 0
    active_features = 0
    lambda_values = []

    # ---- Extract structural info ----
    for term in model.terms:
        # nsp is the number of splines/knots for a term
        nsp = getattr(term, "n_splines", 0) 
        lam = getattr(term, "lam", None)

        # Collect lambdas (linear or spline)
        if lam is not None:
            lam = np.mean(lam) if isinstance(lam, (list, np.ndarray)) else lam
            lambda_values.append(float(lam))

        # Active term?
        if nsp >= 1:
            active_features += 1
            # Count the number of knots
            total_knots += max(nsp, 1)

    # ---- 1. Knot Normalization (Now the Dominant Term) ----
    max_total_knots = max(1, n_features * max_knots_per_feature)
    norm_knots = total_knots / max_total_knots

    # ---- 2. EDOF Normalization ----
    total_edof = model.statistics_.get("edof", 0.0)
    # The maximum possible effective degrees of freedom (DOF) is roughly proportional to the total knots
    max_edof = max(1.0, n_features * (max_knots_per_feature - 1)) 
    norm_edof = float(total_edof) / max_edof

    # ---- 3. Feature Usage Normalization ----
    norm_features = active_features / max(1, n_features)

    # ---- 4. λ Penalty (Increased Slope) ----
    def lambda_penalty(lambda_values):
        if len(lambda_values) == 0:
            return 1.0

        lam = np.exp(np.mean(np.log(np.clip(lambda_values, 1e-12, None))))
        loglam = np.log10(lam)

        # Increased slope (a=3.0) to harshly penalize log10(lambda) < 0
        b = 0      # Midpoint at lambda=1
        a = 3.0    # <-- Steeper penalty than previous revisions

        # small lam → penalty ~ 1 ; large lam → penalty → 0
        penalty = 1 / (1 + np.exp(a * (loglam - b)))

        return float(np.clip(penalty, 0.0, 1.0))

    norm_lambda = lambda_penalty(lambda_values)

    # ---- Combine Penalties (Knot count is now the primary structural penalty) ----
    # Rationale for weights: 
    # 1. Knots (0.50): Strongest pressure for narrow CI.
    # 2. Features (0.35): Essential for sparsity (is the feature even needed?).
    # 3. Lambda (0.10): Secondary pressure for smoothness.
    # 4. EDOF (0.05): Tertiary control, mostly managed by Knots and Lambda.
    penalty = (
        0.50 * norm_knots +       # <-- Major increase in weight
        0.05 * norm_edof +
        0.35 * norm_features +
        0.10 * norm_lambda
    )

    return float(np.clip(penalty, 0.0, 1.0))


def improved_main(seeds: Optional[List[int]] = None, population_size: int = 80, n_generations: int = 80):
    """Run the improved GA once per seed in `seeds` and save per-seed and aggregate md."""
    # Determine seeds from env or function arg
    env_seeds = os.environ.get("SEEDS")
    if env_seeds:
        seeds_list = [int(s.strip()) for s in env_seeds.split(",") if s.strip()]
    elif seeds is not None:
        seeds_list = list(seeds)
    else:
        seeds_list = [42]

    aggregate_rows = []
    aggregate_path = "outputs/improved_results_all_seeds.md"

    # Feature names (California housing)
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                    "Population", "AveOccup", "Latitude", "Longitude"]

    for SEED in seeds_list:
        print(f"\n=== Running improved GA for seed {SEED} ===")
        set_global_seed(SEED)

        # Load data
        dataset = RegressionDataset(default="california", random_state=SEED, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()
        n_features = X_train.shape[1]

        # Setup improved DEAP
        toolbox = setup_improved_deap(n_features, X_train, y_train, X_val, y_val, population_size)
        
        # Run improved GA
        final_pop, logbook, hof = run_improved_ga(toolbox, population_size, n_generations)
        
        pareto_front = sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]

        visualize_pareto_front(pareto_front, f'outputs/improved_pareto_front_seed{SEED}.png')
        best_by_rmse = pick_best_by_rmse(pareto_front)
        best_by_penalty = pick_best_by_penalty(pareto_front)
        
        #knee solution
        knee = pick_knee_solution(final_pop)

        # Prepare three GA candidates: knee, best_by_rmse, best_by_penalty
        best_chrom_knee = deepcopy(knee)
        #best_chrom_knee = greedy_prune(best_chrom_knee, X_train, y_train, X_val, y_val, tol=1e-3)

        best_chrom_rmse = deepcopy(best_by_rmse)
        #best_chrom_rmse = greedy_prune(best_chrom_rmse, X_train, y_train, X_val, y_val, tol=1e-3)

        best_chrom_penalty = deepcopy(best_by_penalty)
        #best_chrom_penalty = greedy_prune(best_chrom_penalty, X_train, y_train, X_val, y_val, tol=1e-3)

        # Build and train final models on combined data (handle possible None builders)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])

        def build_and_fit(chrom):
            gam = ImprovedGAMBuilder.build(chrom)
            if gam is None:
                return None
            try:
                gam.fit(X_combined, y_combined)
                return gam
            except Exception:
                return None

        best_gam_knee = build_and_fit(best_chrom_knee)
        best_gam_rmse = build_and_fit(best_chrom_rmse)
        best_gam_penalty = build_and_fit(best_chrom_penalty)

        # Evaluate on test set (safely handle None)
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

        # Train/evaluate baselines on the same splits
        decision_tree_model, pygam_model, rmse_dt, rmse_gam = test_baselines(
            X_train, X_val, X_test, y_train, y_val, y_test, seed=SEED
        )

        #print(f"Baseline PyGAM Test RMSE: {rmse_gam:.4f}")
        #print(f"Decision Tree Test RMSE: {rmse_dt:.4f}")


        complexity_penalty_baseline = complexity_penalty(pygam_model.model, n_features, max_knots_per_feature=20)
        print(f"Complexity Penalty for baseline: {complexity_penalty_baseline:.4f}")
        complexity_penalty_knee = complexity_penalty(best_gam_knee, n_features, max_knots_per_feature=20)
        complexity_penalty_rmse = complexity_penalty(best_gam_rmse, n_features, max_knots_per_feature=20)
        complexity_penalty_penalty = complexity_penalty(best_gam_penalty, n_features, max_knots_per_feature=20)
        print(f"Complexity Penalty (knee): {complexity_penalty_knee:.4f}")
        print(f"Complexity Penalty (best_by_rmse): {complexity_penalty_rmse:.4f}")
        print(f"Complexity Penalty (best_by_penalty): {complexity_penalty_penalty:.4f}")

        # Visualize (per-seed file)
        models = [best_gam_knee, best_gam_rmse, best_gam_penalty, decision_tree_model, pygam_model]
        model_names = ["GA (knee)", "GA (best_by_rmse)", "GA (best_by_penalty)", "Decision Tree", "Baseline PyGAM"]
        chromosomes = [best_chrom_knee, best_chrom_rmse, best_chrom_penalty, None, None]
        vis_path = f'outputs/improved_feature_effects_seed{SEED}'
        visualize_models(X_test, feature_names, models, model_names, chromosomes, vis_path)

        # Summaries for GA candidates and baseline
        print("Model Structure Summaries:")
        summary_knee = summarize_structure(best_chrom_knee, None, feature_names)
        summary_rmse = summarize_structure(best_chrom_rmse, None, feature_names)
        summary_penalty = summarize_structure(best_chrom_penalty, None, feature_names)
        summary_baseline = summarize_structure(None, pygam_model.model, feature_names)

        print("-- GA (knee):")
        for feature, ftype in summary_knee.items():
            print(f"{feature}: {ftype}")
        print("-- GA (best_by_rmse):")
        for feature, ftype in summary_rmse.items():
            print(f"{feature}: {ftype}")
        print("-- GA (best_by_penalty):")
        for feature, ftype in summary_penalty.items():
            print(f"{feature}: {ftype}")
        print("-- Baseline PyGAM:")
        for feature, ftype in summary_baseline.items():
            print(f"{feature}: {ftype}")

        # Save per-seed results to markdown
        results_path = f"outputs/improved_results_seed{SEED}.md"
        now = datetime.utcnow().isoformat() + "Z"
        md_lines = []
        md_lines.append(f"# Improved Run results — seed {SEED}\n")
        md_lines.append(f"- Date (UTC): {now}\n")
        md_lines.append(f"- Improved GA-GAM (knee) Final Test RMSE: {rmse_knee:.4f}\n")
        md_lines.append(f"- Improved GA-GAM (best_by_rmse) Final Test RMSE: {rmse_best_rmse:.4f}\n")
        md_lines.append(f"- Improved GA-GAM (best_by_penalty) Final Test RMSE: {rmse_best_penalty:.4f}\n")
        md_lines.append(f"- Decision Tree Test RMSE: {rmse_dt:.4f}\n")
        md_lines.append(f"- Baseline PyGAM Test RMSE: {rmse_gam:.4f}\n")
        md_lines.append(f"- Complexity Penalty (baseline): {complexity_penalty_baseline:.4f}\n")
        md_lines.append(f"- Complexity Penalty (knee): {complexity_penalty_knee:.4f}\n")
        md_lines.append(f"- Complexity Penalty (best_by_rmse): {complexity_penalty_rmse:.4f}\n")
        md_lines.append(f"- Complexity Penalty (best_by_penalty): {complexity_penalty_penalty:.4f}\n")
        md_lines.append("\n## Generation Log\n")
        md_lines.append("| Gen | Best RMSE | Average RMSE |\n|---:|---:|---:|\n")
        for entry in logbook:
            # stats produce numpy arrays for multi-objective values; take first element = RMSE
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

        md_lines.append("\n## Best Chromosome (JSON)\n")
        # Save chromosomes JSON for all three GA candidates
        def safe_json(gen):
            try:
                return json.dumps(gen, indent=2)
            except Exception:
                return str(gen)

        md_lines.append("\n## Best Chromosomes (JSON)\n")
        md_lines.append("### GA (knee)\n")
        md_lines.append("```json")
        md_lines.append(safe_json(best_chrom_knee.genes))
        md_lines.append("```")
        md_lines.append("### GA (best_by_rmse)\n")
        md_lines.append("```json")
        md_lines.append(safe_json(best_chrom_rmse.genes))
        md_lines.append("```")
        md_lines.append("### GA (best_by_penalty)\n")
        md_lines.append("```json")
        md_lines.append(safe_json(best_chrom_penalty.genes))
        md_lines.append("```")

        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"Saved results to {results_path}")

        # Append to aggregate rows
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

    # Write aggregate summary
    agg_lines = []
    agg_lines.append("# Improved Aggregate results for seeds\n")
    agg_lines.append("| Seed | GA (knee) RMSE | GA (best_by_rmse) RMSE | GA (best_by_penalty) RMSE | DecisionTree RMSE | PyGAM RMSE | Improvement | Results MD | Visualization |\n")
    agg_lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
    for r in aggregate_rows:
        improvement = r.get('improvement_over_baseline', float('nan'))
        improvement_text = f"{improvement:+.4f}"
        if not np.isnan(improvement) and improvement > 0:
            improvement_text += " Success:"
        else:
            improvement_text += " Failure:"
        agg_lines.append(f"| {r['seed']} | {r.get('rmse_knee', float('nan')):.4f} | {r.get('rmse_best_by_rmse', float('nan')):.4f} | {r.get('rmse_best_by_penalty', float('nan')):.4f} | {r['rmse_dt']:.4f} | {r['rmse_gam']:.4f} | {improvement_text} | {r['results_md']} | {r['vis']} |\n")

    # Calculate average improvement (nan-safe)
    avg_improvement = float(np.nanmean([r.get('improvement_over_baseline', float('nan')) for r in aggregate_rows]))
    agg_lines.append(f"\n**Average improvement over baseline PyGAM: {avg_improvement:+.4f}**\n")
    
    if avg_improvement > 0:
        agg_lines.append("\n**SUCCESS: Improved GA-GAM outperforms baseline PyGAM on average!**\n")
    else:
        agg_lines.append("\n**Suggestion: Try increasing n_generations or adjusting hyperparameters.**\n")

    with open(aggregate_path, "w", encoding="utf-8") as f:
        f.write("\n".join(agg_lines))

    print(f"Saved aggregate results to {aggregate_path}")
    return aggregate_rows

# ------------------------------
# 6. Main Improved Runner
# ------------------------------
if __name__ == "__main__":
    # Run improved GA
    print("=" * 60)
    print("RUNNING IMPROVED GA-GAM")
    print("=" * 60)
    
    improved_results = improved_main(seeds=[42, 7, 123,225,729], population_size=80, n_generations=20)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for result in improved_results:
        seed = result['seed']
        improvement = result['improvement_over_baseline']
        status = "Success" if improvement > 0 else "Failure"
        print(f"Seed {seed}: {status} (Improvement: {improvement:+.4f})")
    
    avg_improvement = np.mean([r['improvement_over_baseline'] for r in improved_results])
    print(f"\nOverall average improvement: {avg_improvement:+.4f}")
    
    if avg_improvement > 0:
        print("SUCCESS: Improved GA-GAM consistently outperforms baseline PyGAM!")
    else:
        print("Not real improvements")