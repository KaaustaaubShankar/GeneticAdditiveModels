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
            weights = [0.2, 0.3, 0.5]  # [none, linear, spline]
            component_type = random.choices(COMPONENT_TYPES, weights=weights)[0]
            
            gene = {
                "type": component_type,
                "knots": random.randint(8, self.max_knots) if component_type == "spline" else None,
                "lambda": random.uniform(0.1, 10.0) if component_type == "spline" else None,
                "scale": random.choice([True, False])
            }
            self.genes.append(gene)

    def mutate(self, mutation_rate=0.15, generation=0, max_generations=100):
        """Adaptive mutation that decreases over time"""
        adaptive_rate = mutation_rate * (1 - generation / max_generations)
        
        for gene in self.genes:
            if random.random() < adaptive_rate:
                if random.random() < 0.7:  # 70% chance to change type
                    gene["type"] = random.choice(COMPONENT_TYPES)
                    if gene["type"] == "spline":
                        gene["knots"] = random.randint(8, self.max_knots)
                        gene["lambda"] = random.uniform(0.1, 10.0)
                    else:
                        gene["knots"] = None
                        gene["lambda"] = None
                else:  # 30% chance to tweak hyperparameters
                    if gene["type"] == "spline":
                        # Small perturbations
                        gene["knots"] = max(4, min(self.max_knots, 
                                                 gene["knots"] + random.randint(-2, 2)))
                        gene["lambda"] = max(0.001, gene["lambda"] * random.uniform(0.8, 1.2))
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
                new_g1["lambda"] = random.uniform(0.1, 10.0)

            if new_g2["type"] == "spline" and new_g2["knots"] is None:
                new_g2["knots"] = random.randint(8, self.max_knots)
                new_g2["lambda"] = random.uniform(0.1, 10.0)

            # Append to the child gene lists
            child1_genes.append(new_g1)
            child2_genes.append(new_g2)

        # Create child chromosomes using the same class
        child1 = self.__class__(self.n_features, self.max_knots)
        child2 = self.__class__(self.n_features, self.max_knots)
        child1.genes = child1_genes
        child2.genes = child2_genes
        return child1, child2

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
                # Use minimal splines + high lambda to mimic linear
                term = s(i, n_splines=2, lam=1000.0)
                
            elif gene["type"] == "spline":
                # Constrain total knots to prevent overfitting
                available_knots = min(gene["knots"], max_total_knots - total_knots)
                if available_knots >= 4:  # Minimum sensible knots
                    term = s(i, n_splines=available_knots, lam=gene["lambda"])
                    total_knots += available_knots
                else:
                    continue  # Skip if not enough knots available
                    
            if terms is None:
                terms = term
            else:
                terms += term
                
        if terms is None:
            # Fallback: use linear terms for all features
            terms = s(0, n_splines=2, lam=1000.0)
            for i in range(1, chromosome.n_features):
                terms += s(i, n_splines=2, lam=1000.0)
                
        gam = LinearGAM(terms)
        return gam

# ------------------------------
# 4. Improved Fitness Evaluator
# ------------------------------
class ImprovedGAMFitnessEvaluator:
    @staticmethod
    def evaluate(chromosome, X_train, y_train, X_val, y_val, 
                 n_repeats=2, cv_folds=3):
        """
        Improved fitness evaluation with cross-validation and AIC-like penalty
        """
        # Use cross-validation for more robust evaluation
        fold_scores = []
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        n_samples = len(X_combined)
        fold_size = n_samples // cv_folds
        
        for fold in range(cv_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
            
            # Create train/val splits
            X_val_fold = X_combined[val_start:val_end]
            X_train_fold = np.delete(X_combined, slice(val_start, val_end), axis=0)
            y_val_fold = y_combined[val_start:val_end]
            y_train_fold = np.delete(y_combined, slice(val_start, val_end))
            
            try:
                gam = ImprovedGAMBuilder.build(chromosome)
                gam.fit(X_train_fold, y_train_fold)
                preds = gam.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
                
                # Calculate AIC-like penalty based on model complexity
                n_splines_total = 0
                for term in gam.terms:
                    if hasattr(term, 'n_splines'):
                        n_splines_total += term.n_splines
                
                # Approximate effective parameters (degrees of freedom)
                effective_params = gam.statistics_['edof']
                
                # AIC-like penalty
                aic_penalty = (2 * effective_params) / len(X_val_fold)
                
                # Final fitness (higher is better)
                fold_score = -rmse - aic_penalty
                fold_scores.append(fold_score)
                
            except Exception as e:
                fold_scores.append(-np.inf)
        
        return np.mean(fold_scores)

# ------------------------------
# 5. Improved GA Setup with DEAP
# ------------------------------
def setup_improved_deap(n_features, X_train, y_train, X_val, y_val, population_size=80):
    # Clean up existing classes to avoid conflicts
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual

    # DEAP creator setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", ImprovedGAMChromosome, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, n_features=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return (ImprovedGAMFitnessEvaluator.evaluate(ind, X_train, y_train, X_val, y_val),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", lambda ind1, ind2: ind1.crossover(ind2, swap_prob=0.3))
    
    def mutate_individual(individual, generation=0, max_generations=100):
        individual.mutate(mutation_rate=0.15, generation=generation, max_generations=max_generations)
        return individual,
    
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=5)

    return toolbox

def run_improved_ga(toolbox, population_size=80, n_generations=80):
    """Run improved GA with better strategy"""
    population = toolbox.population(n=population_size)
    
    # Statistics for tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of fame to keep best individuals
    hof = tools.HallOfFame(1)
    
    # Use eaSimple with custom parameters
    final_pop, logbook = algorithms.eaSimple(
        population, 
        toolbox,
        cxpb=0.7,  # Crossover probability
        mutpb=0.2,  # Mutation probability
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
        dataset = RegressionDataset(default="california", random_state=SEED)
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()
        n_features = X_train.shape[1]

        # Setup improved DEAP
        toolbox = setup_improved_deap(n_features, X_train, y_train, X_val, y_val, population_size)
        
        # Run improved GA
        final_pop, logbook, hof = run_improved_ga(toolbox, population_size, n_generations)
        
        # Get best chromosome from hall of fame
        best_chrom = hof[0]
        
        # Build and train final model on combined data
        best_gam = ImprovedGAMBuilder.build(best_chrom)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        best_gam.fit(X_combined, y_combined)
        
        # Evaluate on test set
        preds = best_gam.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Improved GA-GAM Test RMSE: {rmse_test:.4f}")

        # Train/evaluate baselines on the same splits
        decision_tree_model, pygam_model, rmse_dt, rmse_gam = test_baselines(
            X_train, X_val, X_test, y_train, y_val, y_test, seed=SEED
        )

        print(f"Baseline PyGAM Test RMSE: {rmse_gam:.4f}")
        print(f"Decision Tree Test RMSE: {rmse_dt:.4f}")

        # Visualize (per-seed file)
        models = [best_gam, decision_tree_model, pygam_model]
        model_names = ["Improved GA GAM", "Decision Tree", "Baseline PyGAM"]
        chromosomes = [best_chrom, None, None]
        vis_path = f'outputs/improved_feature_effects_seed{SEED}.png'
        visualize_models(X_test, feature_names, models, model_names, chromosomes, vis_path)

        summary = summarize_structure(best_chrom, pygam_model.model, feature_names)
        print("Model Structure Summary:")
        for feature, ftype in summary.items():
            print(f"{feature}: {ftype}")

        # Save per-seed results to markdown
        results_path = f"outputs/improved_results_seed{SEED}.md"
        now = datetime.utcnow().isoformat() + "Z"
        md_lines = []
        md_lines.append(f"# Improved Run results — seed {SEED}\n")
        md_lines.append(f"- Date (UTC): {now}\n")
        md_lines.append(f"- Improved GA-GAM Final Test RMSE: {rmse_test:.4f}\n")
        md_lines.append(f"- Decision Tree Test RMSE: {rmse_dt:.4f}\n")
        md_lines.append(f"- Baseline PyGAM Test RMSE: {rmse_gam:.4f}\n")
        
        md_lines.append("\n## Generation Log\n")
        md_lines.append("| Gen | Best Fitness | Average Fitness |\n|---:|---:|---:|\n")
        for entry in logbook:
            md_lines.append(f"| {entry['gen']} | {entry['max']:.6f} | {entry['avg']:.6f} |\n")

        md_lines.append("\n## Model Structure Summary\n")
        for feature, ftype in summary.items():
            md_lines.append(f"- **{feature}**: {ftype}\n")

        md_lines.append("\n## Best Chromosome (JSON)\n")
        try:
            genes_json = json.dumps(best_chrom.genes, indent=2)
        except Exception:
            genes_json = str(best_chrom.genes)
        md_lines.append("```json")
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
            "improvement_over_baseline": float(rmse_gam - rmse_test),
            "vis": vis_path,
            "results_md": results_path,
        })

    # Write aggregate summary
    agg_lines = []
    agg_lines.append("# Improved Aggregate results for seeds\n")
    agg_lines.append("| Seed | GA-GAM RMSE | DecisionTree RMSE | PyGAM RMSE | Improvement | Results MD | Visualization |\n")
    agg_lines.append("|---:|---:|---:|---:|---:|---|---|\n")
    for r in aggregate_rows:
        improvement = r['improvement_over_baseline']
        improvement_text = f"{improvement:+.4f}" 
        if improvement > 0:
            improvement_text += " Success:"
        else:
            improvement_text += " Failure:"
        agg_lines.append(f"| {r['seed']} | {r['rmse_ga']:.4f} | {r['rmse_dt']:.4f} | {r['rmse_gam']:.4f} | {improvement_text} | {r['results_md']} | {r['vis']} |\n")
    
    # Calculate average improvement
    avg_improvement = np.mean([r['improvement_over_baseline'] for r in aggregate_rows])
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
    
    improved_results = improved_main(seeds=[42, 7, 123,225,729], population_size=80, n_generations=50)
    
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