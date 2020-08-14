# GraphPlas
GraphPlas: Assembly Graph Assisted Recovery of Plasmidic Contigs from TGS Assemblies

# Usage Instructions

```
usage: GraphPlas.py [-h] --spades-path SPADES_PATH --output OUTPUT
                    [--threads THREADS] [--plots]
                    [--ground-truth GROUND_TRUTH]

GraphPlas Plasmid Detection Using Assembly Graph

optional arguments:
  -h, --help            show this help message and exit
  --spades-path SPADES_PATH, -s SPADES_PATH
                        Assembly path
  --output OUTPUT, -o OUTPUT
                        Contigs file (contigs.fasta)
  --threads THREADS, -t THREADS
                        Thread limit
  --plots               Whether to plot the graph images
  --ground-truth GROUND_TRUTH, -i GROUND_TRUTH
                        Ground truths file (tab separated file with contig_id
                        and truth label on each line). Plasmid label =
                        plasmid, Chromosome label = chromosome
```