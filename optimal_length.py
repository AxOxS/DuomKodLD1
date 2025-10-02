import os, numpy as np, matplotlib.pyplot as plt

from text_analysis import TextAnalysis
from config import TEXTS, RESULT_FOLDER, OPTIMAL_LENGTH_SETTINGS


class OptimalLengthAnalysis:
    def __init__(self):
        self.analysis = TextAnalysis()
    #Analyse the text and print char frequency reliability metrics for their lengths         
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
        
        #Storage for each tested length incremented by step, stability score and entropy for each length 
        lengths = []
        stability_metrics = []
        entropy_list = []
        
        #Loop for testing each length from min, to max or end of texts, incrementing by step
        for length in range(min_length, min(max_length + 1, len(text)), step):
            if length >= len(text):
                break
            
            #Split each length into 3 equal parts and calculate their frequencies separately
            #to check character frequency consistency across different parts of the same text
            part1 = text[:length//3]
            part2 = text[length//3:2*length//3]
            part3 = text[2*length//3:length]
            
            frequencies1 = self.analysis.get_char_frequencies(part1)
            frequencies2 = self.analysis.get_char_frequencies(part2)
            frequencies3 = self.analysis.get_char_frequencies(part3)
            
            #Add all chars that appear in 3 parts in one set without repetition 
            all_symbols = set(frequencies1.keys()) | set(frequencies2.keys()) | set(frequencies3.keys())
            
            differences = []
            #Take the frequencies of each symbol of each part, if symbol doesn't exist in that part - 0.
            for symbol in all_symbols:
                d1 = frequencies1.get(symbol, 0)
                d2 = frequencies2.get(symbol, 0)
                d3 = frequencies3.get(symbol, 0)
                
                avg = (d1 + d2 + d3) / 3
                #Get the variance from average. Square to eliminate posible negative values
                difference = ((d1 - avg)**2 + (d2 - avg)**2 + (d3 - avg)**2) / 3
                differences.append(difference)
            
            #Root mean square deviation. Calculates the average variance across all characters and gets
            #square root for standard deviation, remove the square for unit similarity
            #Lower stability indicator means better reliability, more consistent frequencies
            stability_indicator = np.sqrt(np.mean(differences))
            
            #Calculate Shannon entropy for the text at the current length (randomness/complexity)
            text_entropy = self.analysis.get_char_frequencies(text[:length])
            entropy = self.analysis.calculate_entropy(text_entropy)
            
        
            lengths.append(length)
            stability_metrics.append(stability_indicator)
            entropy_list.append(entropy)
            
            #Show values at an interval of 2500 characters
            if length % (step * 5) == 0:
                print(f"  Analyzed length: {length}, stability: {stability_indicator:.6f}")
        
        return lengths, stability_metrics, entropy_list
    
    
    def visualize_length_analysis(self, lengths: list, stability_indicators: list, 
                                  entropies: list, name: str):
        
        plt.figure(figsize=(12, 6))
        
        # Stability indicator graph
        plt.plot(lengths, stability_indicators, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Text length (characters)')
        plt.ylabel('Stability indicator')
        plt.title(f'Character frequency stability vs text length\n({name})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save graph
        filename = f'length_analysis_{name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(RESULT_FOLDER, filename), dpi=300, bbox_inches='tight')
        plt.show()
    
    
    def analyse_all_texts(self):
        
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
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
        
        return optimal_lengths