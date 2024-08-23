from src.dataset_stats import extract_jsons, print_stat, max_tokens, min_tokens

def main():
    datasets = {'original': ([], [], [])}
    translated_dataset = {'translated_manually': ([], [], []),
                          'translated_Unbabel_TowerInstruct-v0.1_substring_logic': ([], [], [])}
    extract_jsons(datasets, is_translated=False)
    extract_jsons(translated_dataset, is_translated=True)

    print_stat(datasets, max_tokens, "Maximum", "tokens")
    print_stat(datasets, min_tokens, "Minimum", "tokens")
    print_stat(translated_dataset, max_tokens, "Maximum", "tokens")
    print_stat(translated_dataset, min_tokens, "Minimum", "tokens")

if __name__ == '__main__':
    main()
