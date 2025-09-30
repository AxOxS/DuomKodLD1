# Text Analysis System - Data Encoding and Computer Networks

This system performs character frequency analysis and Markov source modeling for various texts.

## Project Objectives

1. **Calculate and compare** alphabet character occurrence probabilities in different types of texts
2. **Model text** with appropriate order Markov source

## Project Structure

```
DuomKodLD1/
├── main.py                 # Main execution script
├── text_analysis.py       # Main character analysis class
├── optimal_length.py      # Optimal text length determination
├── markov_analysis.py     # Markov source analysis
├── tekstai/                # Analyzed texts
│   ├── lt_grozinis.txt     # Lithuanian literary text
│   ├── lt_mokslinis.txt    # Lithuanian scientific text
│   ├── es_grozinis.txt     # Spanish literary text
│   └── es_mokslinis.txt    # Spanish scientific text
├── results/                # Analysis results and graphs
└── README.md              # This instruction
```

## Research Objects

- **Lithuanian Literary Text** (~50,000 characters)
- **Lithuanian Scientific Text** (~50,000 characters)
- **Spanish Literary Text** (~50,000 characters)
- **Spanish Scientific Text** (~50,000 characters)

## System Capabilities

### 1. Optimal Text Length Determination
- Analyzes character frequency stability in different text parts
- Determines minimum text length for reliable frequency estimation
- Visualizes stability changes

### 2. Character Frequency Analysis
- Calculates each character's occurrence probability
- Compares frequencies between different text types
- Creates visual frequency comparisons

### 3. Entropy Calculation
- Calculates Shannon entropy: H = -Σ p(aᵢ) * log₂(p(aᵢ))
- Compares entropies of different texts
- Analyzes information content in texts

### 4. Markov Source Modeling
- Checks zero-order Markov source suitability
- Calculates conditional probabilities
- Performs chi-square test for independence hypothesis
- Visualizes conditional probabilities in heatmap format

## Execution Instructions

### Automatic Execution

Run the main script which will perform all analyses:

```bash
python main.py
```

### Individual Module Execution

**1. Optimal length analysis:**
```bash
python optimal_length.py
```

**2. Character and entropy analysis:**
```bash
python text_analysis.py
```

**3. Markov source analysis:**
```bash
python markov_analysis.py
```

## Results

All analysis results are saved in the `results/` folder:

### Graphs (PNG format)
- `entropy_comparison.png` - Text entropy comparison
- `character_frequencies_*.png` - Character frequency comparisons
- `length_analysis_*.png` - Optimal length analysis for each text
- `conditional_probabilities_*.png` - Conditional probability heatmaps
- `markov_comparison.png` - Chi-square test comparisons


## Methodology Description

### Stability Analysis
1. Text is divided into parts
2. Character frequencies are calculated in each part
3. Standard deviation between parts is calculated
4. Optimal length with smallest deviation is found

### Entropy Calculation
- Shannon entropy formula is used
- Calculated in bits per character
- Different text entropies are compared

### Chi-Square Test
- **H₀**: Characters are independent (zero-order Markov source suitable)
- **H₁**: Characters depend on preceding ones
- **α = 0.05** (significance level)

## Requirements

### Python Packages
- `numpy` - For numerical calculations
- `matplotlib` - For graph creation
- `pandas` - For data processing
- `seaborn` - For advanced visualization
- `scipy` - For statistical tests

### System
- Python 3.7+
- Windows/Linux/macOS
