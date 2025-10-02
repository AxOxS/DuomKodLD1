import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from collections import defaultdict, Counter
from scipy import stats
from base_utils import BaseTextAnalysis
from config import RESULT_FOLDER, MARKOV_SETTINGS


class MarkovAnalysis(BaseTextAnalysis):
    def __init__(self):
        super().__init__()
        self.markov_results = {}
        
    def calculate_conditional_probabilities(self, text: str, order: int = 1):
        """
        Calculates conditional probabilities for Markov source (excluding spaces)
        order: 0 - zero order (independent), 1 - first order, 2 - second order
        """
        # Filter out spaces
        filtered_text = text.replace(' ', '')
        if not filtered_text:
            return {}
            
        if order == 0:
            # Zero order - simple character probabilities
            character_counts = Counter(filtered_text)
            total_count = len(filtered_text)
            return {character: count / total_count 
                   for character, count in character_counts.items()}
        
        # First or higher order
        conditional_counts = defaultdict(lambda: defaultdict(int))
        condition_counts = defaultdict(int)
        
        for i in range(order, len(filtered_text)):
            condition = filtered_text[i-order:i]
            current = filtered_text[i]
            
            conditional_counts[condition][current] += 1
            condition_counts[condition] += 1
        
        # Calculate conditional probabilities
        conditional_probabilities = {}
        for condition in conditional_counts:
            conditional_probabilities[condition] = {}
            if condition_counts[condition] > 0:
                for character, count in conditional_counts[condition].items():
                    conditional_probabilities[condition][character] = count / condition_counts[condition]
        
        return conditional_probabilities
    

    
    def chi_square_test(self, text: str):
        """
        Performs chi-square test to check independence hypothesis
        """
        print(f"\n=== CHI-SQUARE TEST ===")
        
        # Filter out spaces
        filtered_text = text.replace(' ', '')
        if len(filtered_text) < 2:
            print("Not enough characters for chi-square test")
            return 0, 1, 0  # Return defaults
        
        # Take most frequent characters
        character_counts = Counter(filtered_text)
        most_frequent_chars = [s for s, _ in character_counts.most_common(MARKOV_SETTINGS['most_frequent_symbols_count'])]
        
        # Create contingency table
        contingency_table = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(filtered_text)):
            previous = filtered_text[i-1]
            current = filtered_text[i]
            
            if previous in most_frequent_chars and current in most_frequent_chars:
                contingency_table[previous][current] += 1
        
        # Convert to numpy array
        matrix = []
        for previous in most_frequent_chars:
            row = []
            for current in most_frequent_chars:
                row.append(contingency_table[previous][current])
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Perform chi-square test
        chi2, p_value, degrees_of_freedom, expected = stats.chi2_contingency(matrix)
        
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Degrees of freedom: {degrees_of_freedom}")
        
        if p_value < 0.05:
            print("CONCLUSION: Characters are NOT independent (p < 0.05)")
            print("Zero-order Markov source is NOT SUITABLE")
        else:
            print("CONCLUSION: Characters are independent (p >= 0.05)")
            print("Zero-order Markov source is SUITABLE")
        
        return chi2, p_value, degrees_of_freedom
    
    def visualize_conditional_probabilities(self, text: str, name: str):
        """Visualizes conditional probabilities"""
        
        # Filter out spaces and take most frequent characters
        filtered_text = text.replace(' ', '')
        character_counts = Counter(filtered_text)
        most_frequent_chars = [s for s, _ in character_counts.most_common(8) if s.isalpha()]
        
        if len(most_frequent_chars) < 4:
            print(f"Not enough characters for visualization ({len(most_frequent_chars)} < 4)")
            return
        
        # Calculate conditional probabilities
        conditional_probabilities = self.calculate_conditional_probabilities(text, 1)
        
        # Create matrix for visualization
        matrix = np.zeros((len(most_frequent_chars), len(most_frequent_chars)))
        
        for i, previous in enumerate(most_frequent_chars):
            for j, current in enumerate(most_frequent_chars):
                if previous in conditional_probabilities and current in conditional_probabilities[previous]:
                    matrix[i, j] = conditional_probabilities[previous][current]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   xticklabels=most_frequent_chars,
                   yticklabels=most_frequent_chars,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Conditional probability'})
        
        plt.title(f'Conditional probabilities P(current|previous)\n{name}')
        plt.xlabel('Current character')
        plt.ylabel('Previous character')
        
        plt.tight_layout()
        filename = f'conditional_probabilities_{name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(RESULT_FOLDER, filename), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_text(self, name: str):
        """Performs Markov analysis for one text (focuses on zero-order suitability)"""
        if name not in self.texts:
            print(f"Text '{name}' not found!")
            return
        
        text = self.texts[name]
        print(f"\n{'='*60}")
        print(f"MARKOV ANALYSIS: {name}")
        print(f"{'='*60}")
        
        # Chi-square test for independence
        chi2, p_value, degrees_of_freedom = self.chi_square_test(text)
        
        # Visualization
        self.visualize_conditional_probabilities(text, name)
        
        # Save results
        self.markov_results[name] = {
            'chi2_test': {
                'chi2': chi2,
                'p_value': p_value,
                'degrees_of_freedom': degrees_of_freedom
            }
        }
        
        return self.markov_results[name]
    
    def compare_texts(self):
        """Compares Markov analysis results between texts"""
        if len(self.markov_results) < 2:
            print("Not enough texts for comparison!")
            return
        
        print(f"\n{'='*60}")
        print("MARKOV ANALYSIS COMPARISON")
        print(f"{'='*60}")
        
        # Comparison table
        df_data = []
        for name, results in self.markov_results.items():
            chi2_data = results['chi2_test']
            
            df_data.append({
                'Text': name,
                'Chi-square': f"{chi2_data['chi2']:.2f}",
                'p-value': f"{chi2_data['p_value']:.6f}",
                'Independence': 'Yes' if chi2_data['p_value'] >= 0.05 else 'No',
                'Zero-order Markov suitability': 'SUITABLE' if chi2_data['p_value'] >= 0.05 else 'NOT SUITABLE'
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Visual comparison
        self.visualize_comparison()
    
    def visualize_comparison(self):
        """Visualizes text comparison"""
        if len(self.markov_results) < 2:
            return
        
        # p-values comparison
        names = list(self.markov_results.keys())
        p_values = [self.markov_results[p]['chi2_test']['p_value'] for p in names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, p_values, 
                      color=['green' if p >= 0.05 else 'red' for p in p_values],
                      alpha=0.7)
        
        plt.axhline(y=0.05, color='black', linestyle='--', 
                   label='Significance level (α = 0.05)')
        plt.title('Chi-square test p-values comparison')
        plt.ylabel('p-value')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Add values on bars
        for bar, p in zip(bars, p_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{p:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = 'markov_comparison.png'
        plt.savefig(os.path.join(RESULT_FOLDER, filename), dpi=300, bbox_inches='tight')
        plt.show()
