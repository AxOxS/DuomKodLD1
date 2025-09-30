import os, math, numpy as np, matplotlib.pyplot as plt

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from base_utils import BaseTextAnalysis
from config import RESULT_FOLDER, OPTIMAL_LENGTH_SETTINGS

class TextAnalysis(BaseTextAnalysis):
    def __init__(self):
        super().__init__()
        self.character_frequencies = {}
        self.entropies = {}
        
    def get_char_frequencies(self, text: str) -> Dict[str, float]:
        """Calculates character occurrence frequencies in text"""
        if not text:
            return {}
        
        # Count each character occurrences
        character_counts = Counter(text)
        total_character_count = len(text)
        
        # Calculate probabilities (frequencies)
        frequencies = {}
        for character, count in character_counts.items():
            frequencies[character] = count / total_character_count
            
        return frequencies
    
    def find_optimal_text_length(self, text: str, min_length: int = None, 
                                 max_length: int = None, step: int = None) -> Tuple[int, List[int], List[float]]:
        """
        Finds optimal text length that ensures stable character frequencies
        Returns: (optimal_length, lengths, stability_indicators)
        """
        # Use configuration values as defaults
        if min_length is None:
            min_length = OPTIMAL_LENGTH_SETTINGS['min_length']
        if max_length is None:
            max_length = OPTIMAL_LENGTH_SETTINGS['max_length']
        if step is None:
            step = OPTIMAL_LENGTH_SETTINGS['step']
            
        lengths = []
        stability_indicators = []
        
        # Analyze different text lengths
        for length in range(min_length, min(max_length + 1, len(text)), step):
            if length >= len(text):
                break
                
            # Split text into two parts and calculate frequencies
            part1 = text[:length//2]
            part2 = text[length//2:length]
            
            frequencies1 = self.get_char_frequencies(part1)
            frequencies2 = self.get_char_frequencies(part2)
            
            # Calculate stability indicator (root mean square deviation)
            all_characters = set(frequencies1.keys()) | set(frequencies2.keys())
            differences = []
            
            for character in all_characters:
                freq1 = frequencies1.get(character, 0)
                freq2 = frequencies2.get(character, 0)
                differences.append((freq1 - freq2) ** 2)
            
            stability_indicator = math.sqrt(sum(differences) / len(differences))
            
            lengths.append(length)
            stability_indicators.append(stability_indicator)
        
        # Find optimal length (where stability indicator is minimal)
        if stability_indicators:
            min_idx = stability_indicators.index(min(stability_indicators))
            optimal_length = lengths[min_idx]
        else:
            optimal_length = min_length
            
        return optimal_length, lengths, stability_indicators
    
    def calculate_entropy(self, frequencies: Dict[str, float]) -> float:
        """Calculates text entropy according to H = -Σ p(ai) * log2(p(ai))"""
        entropy = 0.0
        for probability in frequencies.values():
            if probability > 0:  # log(0) is impossible
                entropy -= probability * math.log2(probability)
        return entropy
    
    def check_markov_source(self, text: str, character: str) -> Dict[str, float]:
        """
        Checks if character occurrence probability depends on preceding characters
        Returns conditional probabilities dictionary
        """
        if len(text) < 2:
            return {}
        
        # Calculate conditional probabilities P(character | preceding)
        conditional_counts = defaultdict(lambda: defaultdict(int))
        preceding_counts = defaultdict(int)
        
        for i in range(1, len(text)):
            preceding = text[i-1]
            current = text[i]
            
            conditional_counts[preceding][current] += 1
            preceding_counts[preceding] += 1
        
        # Calculate conditional probabilities
        conditional_probabilities = {}
        for preceding in conditional_counts:
            if preceding_counts[preceding] > 0:
                if character in conditional_counts[preceding]:
                    conditional_probabilities[preceding] = (
                        conditional_counts[preceding][character] / 
                        preceding_counts[preceding]
                    )
                else:
                    conditional_probabilities[preceding] = 0.0
        
        return conditional_probabilities
    
    def analyse_all(self, text_name: str):
        """Performs complete analysis for one text"""
        if text_name not in self.texts:
            print(f"Text '{text_name}' not found!")
            return
        
        text = self.texts[text_name]
        print(f"\n=== ANALYZING: {text_name} ===")
        print(f"Text length: {len(text)} characters")
        
        # 1. Find optimal text length
        print("\n1. Optimal text length search...")
        optimal_length, lengths, stability_indicators = self.find_optimal_text_length(text)
        print(f"Optimal text length: {optimal_length} characters")
        
        # Use optimal text length for further analysis
        analyzed_text = text[:optimal_length]
        
        # 2. Calculate character frequencies  
        print("\n2. Character frequency calculation...")
        frequencies = self.get_char_frequencies(analyzed_text)
        self.character_frequencies[text_name] = frequencies
        
        # Output most frequent characters
        most_frequent = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        print("10 most frequent characters:")
        for character, frequency in most_frequent:
            char_repr = repr(character) if character in [' ', '\n', '\t'] else character
            print(f"  {char_repr}: {frequency:.4f}")
        
        # 3. Calculate entropy
        print("\n3. Entropy calculation...")
        entropy = self.calculate_entropy(frequencies)
        self.entropies[text_name] = entropy
        print(f"Entropy: {entropy:.4f} bits/character")
        
        # 4. Investigate Markov source suitability
        print("\n4. Markov source investigation...")
        # Take several most frequent characters
        most_frequent_chars = [s[0] for s in most_frequent[:5]]
        
        for character in most_frequent_chars:
            conditional_probs = self.check_markov_source(analyzed_text, character)
            if conditional_probs:
                prob_values = list(conditional_probs.values())
                if len(prob_values) > 1:
                    std_deviation = np.std(prob_values)
                    char_repr = repr(character) if character in [' ', '\n', '\t'] else character
                    print(f"  Character {char_repr}: conditional probabilities standard deviation = {std_deviation:.4f}")
        
        return {
            'optimal_length': optimal_length,
            'lengths': lengths,
            'stability_indicators': stability_indicators,
            'frequencies': frequencies,
            'entropy': entropy
        }
    
    def visualize_results(self):
        """Creates graphs for results visualization"""
        print("\n=== RESULTS VISUALIZATION ===")
        
        # Set font size for graph titles
        plt.rcParams['font.size'] = 10
        
        # 1. Entropy comparison
        if len(self.entropies) > 1:
            plt.figure(figsize=(10, 6))
            
            names = list(self.entropies.keys())
            entropies = list(self.entropies.values())
            
            bars = plt.bar(names, entropies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(names)])
            plt.title('Text Entropy Comparison')
            plt.ylabel('Entropy (bits/character)')
            plt.xticks(rotation=45, ha='right')
            
            # Add values on top of bars
            for bar, entropy in zip(bars, entropies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{entropy:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_FOLDER, 'entropy_comparison.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Character frequency comparison
        if len(self.character_frequencies) >= 2:
            self._compare_character_frequencies()
    
    def _compare_character_frequencies(self):
        """Compares character frequencies between different texts"""
        text_names = list(self.character_frequencies.keys())
        
        # Take pairs of texts for comparison
        for i in range(0, len(text_names), 2):
            if i + 1 < len(text_names):
                name1 = text_names[i]
                name2 = text_names[i + 1]
                
                frequencies1 = self.character_frequencies[name1]
                frequencies2 = self.character_frequencies[name2]
                
                # Find common characters
                common_chars = set(frequencies1.keys()) & set(frequencies2.keys())
                common_chars = [s for s in common_chars if s.isalpha()]  # Only letters
                
                if len(common_chars) >= 10:
                    # Take 15 most frequent common characters
                    common_freqs1 = {s: frequencies1[s] for s in common_chars}
                    most_frequent_common = sorted(common_freqs1.items(), key=lambda x: x[1], reverse=True)[:15]
                    
                    characters = [s[0] for s in most_frequent_common]
                    freqs_1 = [frequencies1[s] for s in characters]
                    freqs_2 = [frequencies2[s] for s in characters]
                    
                    x = np.arange(len(characters))
                    width = 0.35
                    
                    plt.figure(figsize=(12, 6))
                    bars1 = plt.bar(x - width/2, freqs_1, width, label=name1, alpha=0.8)
                    bars2 = plt.bar(x + width/2, freqs_2, width, label=name2, alpha=0.8)
                    
                    plt.title(f'Character Frequency Comparison: {name1} vs {name2}')
                    plt.xlabel('Characters')
                    plt.ylabel('Frequency')
                    plt.xticks(x, characters)
                    plt.legend()
                    plt.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    filename = f'character_frequencies_{name1}_vs_{name2}.png'
                    plt.savefig(os.path.join(RESULT_FOLDER, filename), dpi=300, bbox_inches='tight')
                    plt.show()

