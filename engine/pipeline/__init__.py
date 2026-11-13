"""CPU thread cap for every `python -m pipeline.*` run (runs before numpy/torch are imported).

The container sees 256 CPUs, so PyTorch/OpenMP default to 128 threads, but its cgroup quota is only
~6.8 CPUs (/sys/fs/cgroup/cpu.max). 128 threads burn that quota in a few ms and the kernel freezes the
whole process for the rest of each 100 ms period, GPU kernel launches included: WHAM extraction on a
150-frame clip took 143 s with the default and 33-35 s with 1-12 threads. Subprocesses (WHAM in its
own venv) inherit the setting. Values already set in the environment win.
"""
import os as _os

PIPELINE_THREADS = '6'
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    _os.environ.setdefault(_k, PIPELINE_THREADS)
