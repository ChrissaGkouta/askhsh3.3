import subprocess
import os
import re

# Σταθεροί συνολικοί πόροι (Total Cores)
TOTAL_CORES = 8 
N = 4096
SPARSITY = 0.95
ITERS = 20

# Συνδυασμοί (MPI Ranks, OpenMP Threads) ώστε Ranks * Threads = TOTAL_CORES
CONFIGS = [
    (8, 1), # Pure MPI
    (4, 2), # Hybrid
    (2, 4), # Hybrid
    (1, 8)  # Almost pure OpenMP (μέσα σε 1 MPI rank)
]

def compile_code():
    subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL)
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)

def run_experiments():
    print(f"{'Ranks':<6} | {'Threads':<8} | {'Time (Calc)':<12} | {'Description'}")
    print("-" * 50)

    for ranks, threads in CONFIGS:
        # Ρύθμιση περιβάλλοντος για OpenMP
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        
        # Εντολή εκτέλεσης
        # ΣΗΜΕΙΩΣΗ: Προσθέτουμε --oversubscribe αν τρέχουμε σε μηχάνημα με λίγους πυρήνες
        # και --map-by node:PE=X για σωστή κατανομή σε clusters (advanced), 
        # αλλά για απλή εργαστηριακή άσκηση αρκεί το βασικό:
        cmd = ["mpirun", "--bind-to", "none", "-np", str(ranks), "./hybrid_spmv", str(N), str(SPARSITY), str(ITERS)]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            
            # Parsing αποτελέσματος
            match = re.search(r"Time_Hybrid_Calc:\s+([0-9\.]+)", res.stdout)
            time_val = float(match.group(1)) if match else 0.0
            
            desc = "Pure MPI" if threads == 1 else ("Pure OMP" if ranks == 1 else "Hybrid")
            print(f"{ranks:<6} | {threads:<8} | {time_val:.6f}     | {desc}")
            
        except subprocess.CalledProcessError as e:
            print(f"{ranks:<6} | {threads:<8} | FAILED       | Check MPI setup")

if __name__ == "__main__":
    compile_code()
    run_experiments()