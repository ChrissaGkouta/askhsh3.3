import subprocess
import os
import re

# --- ΑΛΛΑΓΗ: Τρέχουμε μόνο 4 συνολικά (όσοι οι φυσικοί πυρήνες) ---
# Έτσι αποφεύγουμε το μπούκωμα (oversubscription) και βλέπουμε την πραγματική ταχύτητα
TOTAL_CORES = 4 
N = 4096
SPARSITY = 0.95
ITERS = 20

# Συνδυασμοί που αθροίζουν σε 4
CONFIGS = [
    (4, 1), # Pure MPI (4 procs)
    (2, 2), # Hybrid (2 procs x 2 threads)
    (1, 4)  # Pure OMP (1 proc x 4 threads)
]

def compile_code():
    subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL)
    subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)

def run_experiments():
    print(f"{'Ranks':<6} | {'Threads':<8} | {'Time (Calc)':<12} | {'Description'}")
    print("-" * 55)

    for ranks, threads in CONFIGS:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        
        # Αφαιρούμε το --oversubscribe γιατί τώρα είμαστε εντός ορίων (4 cores)
        # Κρατάμε το --bind-to none για ασφάλεια
        cmd = [
            "mpirun", 
            "--bind-to", "none",
            "-np", str(ranks), 
            "./hybrid_spmv", 
            str(N), str(SPARSITY), str(ITERS)
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
            match = re.search(r"Time_Hybrid_Calc:\s+([0-9\.]+)", res.stdout)
            time_val = float(match.group(1)) if match else 0.0
            
            desc = "Pure MPI" if threads == 1 else ("Pure OMP" if ranks == 1 else "Hybrid")
            print(f"{ranks:<6} | {threads:<8} | {time_val:.6f}     | {desc}")
            
        except subprocess.CalledProcessError:
            print(f"{ranks:<6} | {threads:<8} | FAILED       | Check setup")

if __name__ == "__main__":
    compile_code()
    run_experiments()