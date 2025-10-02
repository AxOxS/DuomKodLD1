from text_analysis import TextAnalysis
from optimal_length import OptimalLengthAnalysis  
from markov_analysis import MarkovAnalysis
from config import TEXTS, get_work_folder, make_result_folder


def main():

    print("="*60)
    print("TEXT ANALYSIS")
    print("Character frequency and Markov analysis")
    print("="*60)
    
    get_work_folder()
    make_result_folder()
    
    # 1. OPTIMAL LENGTH ANALYSIS
    print("\n 1. OPTIMAL TEXT LENGTH DETECTION")
    print("-" * 40)
    
    try:
        optimal_analysis = OptimalLengthAnalysis()
        optimal_lengths = optimal_analysis.analyse_all_texts()
    except Exception as e:
        print(f"Optimal length analysis error: {e}")
        optimal_analysis = None
    
    # 2. MAIN CHARACTER ANALYSIS
    print("\n 2. CHARACTER FREQUENCY AND ENTROPY ANALYSIS")
    print("-" * 40)
    
    try:
        analysis = TextAnalysis()
        
        analysis.get_all_texts(TEXTS)
        
        results = {}
        for name in analysis.texts.keys():
            results[name] = analysis.analyse_all(name)
        
        analysis.visualize_results()
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        analysis = None
    
    # 3. MARKOV ANALYSIS
    print("\n 3. MARKOV MODELING")
    print("-" * 40)
    
    try:
        markov_analysis = MarkovAnalysis()
        
        markov_analysis.get_all_texts(TEXTS)
        
        for name in markov_analysis.texts.keys():
            markov_analysis.analyze_text(name)
        
        markov_analysis.compare_texts()
        
    except Exception as e:
        print(f"Markov analysis error: {e}")
        markov_analysis = None
    
    print(f"\n ANALYSIS COMPLETED!")
    print(f"All results saved in the results folder")
    print("="*60)


if __name__ == "__main__":
    main()
