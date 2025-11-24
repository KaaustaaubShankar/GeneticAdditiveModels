# ------------------------------
# Imports
# ------------------------------
import random
import numpy as np
from copy import deepcopy
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM, s
from deap import base, creator, tools, algorithms
from baselines import test_baselines
from utils.dataset import RegressionDataset
from utils.visualization import visualize_models
from utils.repro import set_global_seed
import os
import json
from datetime import datetime
from typing import List, Optional

os.makedirs("outputs", exist_ok=True)
# ------------------------------
# 2. GAM Chromosome
# ------------------------------
COMPONENT_TYPES = ["none", "linear", "spline"]

class GAMChromosome:
    """
    Chromosome encoding for univariate regression GAM.
    Each feature is a gene with component type and hyperparameters.
    """
    def __init__(self, n_features, max_knots=10):
        self.genes = []
        self.n_features = n_features
        self.max_knots = max_knots
        self._initialize_random()

    def _initialize_random(self):
        self.genes = []
        for _ in range(self.n_features):
            component_type = random.choice(COMPONENT_TYPES)
            gene = {
                "type": component_type,
                "knots": random.randint(4, self.max_knots) if component_type == "spline" else None,
                "lambda": random.uniform(0.001, 1.0) if component_type == "spline" else None,
                "scale": random.choice([True, False])
            }
            self.genes.append(gene)

    def mutate(self, mutation_rate=0.2):
        for gene in self.genes:
            if random.random() < mutation_rate:
                # Randomly change component type
                gene["type"] = random.choice(COMPONENT_TYPES)
                if gene["type"] == "spline":
                    gene["knots"] = random.randint(4, self.max_knots)
                    gene["lambda"] = random.uniform(0.001, 1.0)
                else:
                    gene["knots"] = None
                    gene["lambda"] = None
                gene["scale"] = random.choice([True, False])

    def crossover(self, other, swap_prob=0.5):
        """
        Uniform crossover at the gene *and* hyperparameter level.
        Each gene field (type, knots, lambda, scale) can be swapped independently.
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
                new_g1["knots"] = random.randint(4, self.max_knots)
                new_g1["lambda"] = random.uniform(0.001, 1.0)

            if new_g2["type"] == "spline" and new_g2["knots"] is None:
                new_g2["knots"] = random.randint(4, self.max_knots)
                new_g2["lambda"] = random.uniform(0.001, 1.0)

            # Append to the child gene lists
            child1_genes.append(new_g1)
            child2_genes.append(new_g2)

        # Create child chromosomes
        child1 = GAMChromosome(self.n_features)
        child2 = GAMChromosome(self.n_features)
        child1.genes = child1_genes
        child2.genes = child2_genes
        return child1, child2

# ------------------------------
# 3. GAM Builder
# ------------------------------
class GAMBuilder:
    @staticmethod
    def build(chromosome):
        terms = None
        for i, gene in enumerate(chromosome.genes):
            if gene["type"] == "none":
                continue
            elif gene["type"] == "linear":
                # Use minimal splines + high lambda to mimic linear
                term = s(i, n_splines=4, lam=1e5)
            elif gene["type"] == "spline":
                term = s(i, n_splines=gene["knots"], lam=gene["lambda"])
            if terms is None:
                terms = term
            else:
                terms += term
        if terms is None:
            # No active features, fallback to simple constant
            gam = LinearGAM()
        else:
            gam = LinearGAM(terms)
        return gam

# ------------------------------
# 4. Fitness Evaluator
# ------------------------------
class GAMFitnessEvaluator:
    @staticmethod
    def evaluate(chromosome, X_train, y_train, X_val, y_val, complexity_penalty=0.01, n_repeats=3):
        fitness_values = []
        for _ in range(n_repeats):
            gam = GAMBuilder.build(chromosome)
            try:
                gam.fit(X_train, y_train)
                preds = gam.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                # penalize number of active features
                active_features = sum(1 for g in chromosome.genes if g["type"] != "none")
                fitness_values.append(-rmse - complexity_penalty * active_features)
            except Exception:
                fitness_values.append(-np.inf)  # fallback for numerical issues
        # Return average fitness over n_repeats
        avg_fitness = np.mean(fitness_values)
        return avg_fitness



# ------------------------------
# 5. GA Setup with DEAP
# ------------------------------
def setup_deap(n_features, X_train, y_train, X_val, y_val, population_size=300):
    # DEAP creator should only create classes once; guard against re-creation
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", GAMChromosome, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, n_features=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return (GAMFitnessEvaluator.evaluate(ind, X_train, y_train, X_val, y_val),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", lambda ind1, ind2: ind1.crossover(ind2))
    toolbox.register("mutate", lambda ind: ind.mutate(mutation_rate=0.2))
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

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


def main(seeds: Optional[List[int]] = None, population_size: int = 300):
    """Run the GA once per seed in `seeds` and save per-seed and aggregate md.

    - If `SEEDS` env var is provided it will be parsed as comma-separated ints.
    - Otherwise `seeds` or default [42] will be used.
    """
    # Determine seeds from env or function arg
    env_seeds = os.environ.get("SEEDS")
    if env_seeds:
        seeds_list = [int(s.strip()) for s in env_seeds.split(",") if s.strip()]
    elif seeds is not None:
        seeds_list = list(seeds)
    else:
        seeds_list = [42]

    aggregate_rows = []
    aggregate_path = "outputs/results_all_seeds.md"

    # Feature names (California housing)
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                    "Population", "AveOccup", "Latitude", "Longitude"]

    for SEED in seeds_list:
        print(f"\n=== Running seed {SEED} ===")
        set_global_seed(SEED)

        # Load data
        dataset = RegressionDataset(default="california", random_state=SEED)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()
        n_features = X_train.shape[1]

        # Setup DEAP
        toolbox = setup_deap(n_features, X_train, y_train, X_val, y_val, population_size=population_size)
        population = toolbox.population(n=population_size)

        # Run GA
        NGEN = 100
        gen_logs = []
        for gen in range(NGEN):
            # Evaluate fitness
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = list(map(deepcopy, offspring))

            # Crossover
            for i in range(0, len(offspring), 2):
                if i+1 < len(offspring):
                    child1, child2 = offspring[i].crossover(offspring[i+1])
                    offspring[i].genes, offspring[i+1].genes = child1.genes, child2.genes

            # Mutation
            for ind in offspring:
                ind.mutate(mutation_rate=0.2)

            population[:] = offspring

            # Logging best fitness
            best = max(population, key=lambda ind: ind.fitness.values)
            best_f = float(best.fitness.values[0]) if best.fitness.values is not None else float("nan")
            gen_logs.append((gen + 1, best_f))
            print(f"Gen {gen+1} Best fitness: {best_f:.4f}")

        # Build final GAM from best chromosome
        best_chrom = max(population, key=lambda ind: ind.fitness.values)
        best_gam = GAMBuilder.build(best_chrom)
        best_gam.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
        preds = best_gam.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Final Test RMSE: {rmse_test:.4f}")

        # Train/evaluate baselines on the same splits so results are comparable
        decision_tree_model, pygam_model, rmse_dt, rmse_gam = test_baselines(
            X_train, X_val, X_test, y_train, y_val, y_test, seed=SEED
        )

        # Visualize (per-seed file)
        models = [best_gam, decision_tree_model, pygam_model]
        model_names = ["GA GAM", "Decision Tree", "Baseline PyGAM"]
        chromosomes = [best_chrom, None, None]
        vis_path = f'outputs/feature_effects_seed{SEED}.png'
        visualize_models(X_test, feature_names, models, model_names, chromosomes, vis_path)

        summary = summarize_structure(best_chrom, pygam_model.model, feature_names)
        print("Model Structure Summary:")
        for feature, ftype in summary.items():
            print(f"{feature}: {ftype}")

        # Save per-seed results to markdown
        results_path = f"outputs/results_seed{SEED}.md"
        now = datetime.utcnow().isoformat() + "Z"
        md_lines = []
        md_lines.append(f"# Run results — seed {SEED}\n")
        md_lines.append(f"- Date (UTC): {now}\n")
        md_lines.append(f"- GA-GAM Final Test RMSE: {rmse_test:.4f}\n")
        md_lines.append(f"- Decision Tree Test RMSE: {rmse_dt:.4f}\n")
        md_lines.append(f"- Baseline PyGAM Test RMSE: {rmse_gam:.4f}\n")
        md_lines.append("\n## Generation Log\n")
        md_lines.append("| Gen | Best Fitness |\n|---:|---:|\n")
        for g, f in gen_logs:
            md_lines.append(f"| {g} | {f:.6f} |\n")

        md_lines.append("\n## Model Structure Summary\n")
        for feature, ftype in summary.items():
            md_lines.append(f"- **{feature}**: {ftype}\n")

        md_lines.append("\n## Best Chromosome (JSON)\n")
        try:
            genes_json = json.dumps(best_chrom.genes, indent=2)
        except Exception:
            genes_json = str(best_chrom.genes)
        md_lines.append("```")
        md_lines.append(genes_json)
        md_lines.append("```")

        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"Saved results to {results_path}")

        # Append to aggregate rows
        aggregate_rows.append({
            "seed": SEED,
            "rmse_ga": float(rmse_test),
            "rmse_dt": float(rmse_dt),
            "rmse_gam": float(rmse_gam),
            "vis": vis_path,
            "results_md": results_path,
        })

    # Write aggregate summary
    agg_lines = []
    agg_lines.append("# Aggregate results for seeds\n")
    agg_lines.append("| Seed | GA-GAM RMSE | DecisionTree RMSE | PyGAM RMSE | Results MD | Visualization |\n")
    agg_lines.append("|---:|---:|---:|---:|---|---|\n")
    for r in aggregate_rows:
        agg_lines.append(f"| {r['seed']} | {r['rmse_ga']:.4f} | {r['rmse_dt']:.4f} | {r['rmse_gam']:.4f} | {r['results_md']} | {r['vis']} |\n")

    with open(aggregate_path, "w", encoding="utf-8") as f:
        f.write("\n".join(agg_lines))

    print(f"Saved aggregate results to {aggregate_path}")
# ------------------------------
# 6. Main GA Runner
# ------------------------------
if __name__ == "__main__":

    main(seeds = [42, 7, 123])  # Example seeds to run