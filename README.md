# GraphPlas
GraphPlas: Assembly Graph Assisted Recovery of Plasmidic Contigs from NGS Assemblies

# Usage Instructions

```
usage: GraphPlas [-h] --spades-path SPADES_PATH --classifier CLASSIFIER
                 --classification-file CLASSIFICATION_FILE --output OUTPUT
                 [--threads THREADS] [--min_contig_length MIN_CONTIG_LENGTH]
                 [--plots] [--version] [--ground-truth GROUND_TRUTH]

GraphPlas Plasmid Detection Using Assembly Graph

optional arguments:
  -h, --help            show this help message and exit
  --spades-path SPADES_PATH, -s SPADES_PATH
                        Assembly path
  --classifier CLASSIFIER, -c CLASSIFIER
                        Classifier [pc or pf] default=pc
  --classification-file CLASSIFICATION_FILE, -f CLASSIFICATION_FILE
                        Path to classification file
  --output OUTPUT, -o OUTPUT
                        Contigs file (contigs.fasta)
  --threads THREADS, -t THREADS
                        Thread limit
  --min_contig_length MIN_CONTIG_LENGTH, -m MIN_CONTIG_LENGTH
                        Minimum length of contigs to consider
  --plots               Whether to plot the graph images
  --version, -v         Show version.
  --ground-truth GROUND_TRUTH, -i GROUND_TRUTH
                        Ground truths file (tab separated file with contig_id
                        and truth label on each line). Plasmid label =
                        plasmid, Chromosome label = chromosome
```

## Notes

* The program can get significantly slow if the assembly has many small contigs, disconnected from classified contigs. The minimum contig length can be adjusted using `--min_contig_length` parameter. We now set it at 500 by default. 
* The label propagation algorithm in use is from `https://github.com/ZwEin27/python-labelpropagation` which eliminates the need for matrix inversion of the original algorithm.