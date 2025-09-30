import os, numpy as np, matplotlib.pyplot as plt

from text_analysis import TextAnalysis
from config import TEXTS, RESULT_FOLDER, OPTIMAL_LENGTH_SETTINGS


class OptimalLengthAnalysis:
    def __init__(self):
        self.analysis = TextAnalysis()
        
    def detailed_length_analysis(self, text: str, name: str, 
                            min_length: int = None, max_length: int = None, 
                            step: int = None):

        if min_length is None:
            min_length = OPTIMAL_LENGTH_SETTINGS['min_length']
        if max_length is None:
            max_length = OPTIMAL_LENGTH_SETTINGS['max_length']
        if step is None:
            step = OPTIMAL_LENGTH_SETTINGS['step']

        print(f"\n=== DETAILED LENGTH ANALYSIS: {name} ===")
        
        lengths = []
        stability_metrics = []
        entropy_list = []
        
        for length in range(min_length, min(max_length + 1, len(text)), step):
            if length >= len(text):
                break
            
            part1 = text[:length//3]
            part2 = text[length//3:2*length//3]
            part3 = text[2*length//3:length]
            
            frequencies1 = self.analysis.get_char_frequencies(part1)
            frequencies2 = self.analysis.get_char_frequencies(part2)
            frequencies3 = self.analysis.get_char_frequencies(part3)
            
            all_symbols = set(frequencies1.keys()) | set(frequencies2.keys()) | set(frequencies3.keys())
            
            differences = []
            for symbol in all_symbols:
                d1 = frequencies1.get(symbol, 0)
                d2 = frequencies2.get(symbol, 0)
                d3 = frequencies3.get(symbol, 0)
                
                avg = (d1 + d2 + d3) / 3
                difference = ((d1 - avg)**2 + (d2 - avg)**2 + (d3 - avg)**2) / 3
                differences.append(difference)
            
            stability_indicator = np.sqrt(np.mean(differences))
            
            text_entropy = self.analysis.get_char_frequencies(text[:length])
            entropy = self.analysis.calculate_entropy(text_entropy)
            
            lengths.append(length)
            stability_metrics.append(stability_indicator)
            entropy_list.append(entropy)
            
            if length % (step * 5) == 0:
                print(f"  Analyzed length: {length}, stability: {stability_indicator:.6f}")
        
        return lengths, stability_metrics, entropy_list
    
    
    def visualize_length_analysis(self, lengths: list, stability_indicators: list, 
                                  entropies: list, name: str):
        """Visualizes length analysis results"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Stability indicator graph
        ax1.plot(lengths, stability_indicators, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Text length (characters)')
        ax1.set_ylabel('Stability indicator')
        ax1.set_title(f'Character frequency stability vs text length\n({name})')
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal point
        min_idx = stability_indicators.index(min(stability_indicators))
        optimal_length = lengths[min_idx]
        ax1.axvline(x=optimal_length, color='red', linestyle='--', 
                   label=f'Optimal length: {optimal_length}')
        ax1.legend()
        
        # 2. Entropy graph
        ax2.plot(lengths, entropies, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Text length (characters)')
        ax2.set_ylabel('Entropy (bits/character)')
        ax2.set_title(f'Entropy change vs text length\n({name})')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=optimal_length, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save graph
        filename = f'length_analysis_{name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(RESULT_FOLDER, filename), dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_length
    
    def find_stability_threshold(self, stability_indicators: list, percentage: float = 0.05):
        """
        Finds point where stability indicator drops below certain threshold
        and doesn't rise above it anymore
        """
        min_indicator = min(stability_indicators)
        threshold = min_indicator * (1 + percentage)
        
        # Look for point from which indicator remains below threshold
        for i in range(len(stability_indicators)):
            if stability_indicators[i] <= threshold:
                # Check if indicator won't rise above threshold later
                remaining = stability_indicators[i:]
                if max(remaining) <= threshold * 1.2:  # 20% tolerance
                    return i
        
        return len(stability_indicators) - 1
    
    def analyse_all_texts(self):
        """Analyzes optimal lengths for all texts"""
        
        optimal_lengths = {}
        
        for name, filename in TEXTS.items():
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                print(f"\nAnalyzing: {name}")
                print(f"Total text length: {len(text)} characters")
                
                # Perform detailed analysis
                lengths, stability_indicators, entropies = self.detailed_length_analysis(
                    text, name
                )
                
                # Visualize and find optimal length
                optimal_length = self.visualize_length_analysis(
                    lengths, stability_indicators, entropies, name
                )
                
                optimal_lengths[name] = optimal_length
                
                # Additional analysis - stability threshold
                threshold_index = self.find_stability_threshold(stability_indicators)
                threshold_length = lengths[threshold_index] if threshold_index < len(lengths) else lengths[-1]
                
                print(f"Optimal length (min. stability): {optimal_length}")
                print(f"Stability threshold reached: {threshold_length}")
                print(f"Final stability indicator: {stability_indicators[lengths.index(optimal_length)]:.6f}")
                
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
        
        # Results summary
        self.print_length_summary(optimal_lengths)
        
        return optimal_lengths
    
    def print_length_summary(self, optimal_lengths: dict):
        """Prints optimal lengths summary"""
        print("\n" + "="*50)
        print("OPTIMAL TEXT LENGTHS SUMMARY")
        print("="*50)
        
        for name, length in optimal_lengths.items():
            print(f"{name:25}: {length:5} characters")
        
        if len(optimal_lengths) > 1:
            average = sum(optimal_lengths.values()) / len(optimal_lengths)
            print(f"{'Average':25}: {average:5.0f} characters")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 30)
        if optimal_lengths:
            max_length = max(optimal_lengths.values())
            print(f"Recommended minimum text length for analysis: {max_length} characters")
            print(f"This will ensure stable character frequencies across all texts.")