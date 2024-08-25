# Transformer configuration
transformer_params = {'n_layers': 1,
                      'n_heads': 8,
                      'dropout': 0.2}

# Clinical metadata
continuous_features = ['Age', 'NIHSS at admission', 'mRS at admission']
categorical_features = {'names': ['Sex', 'Atrial fibrillation', 'Hypertension', 'Diabetes'], 'categories': [2, 2, 2, 2]}

# Training configuration
train_params = {'n_epochs': 125,
                'learning_rate': 0.0005}

# Data generator parameters
params = {'imagePath': '/media/kimberly/DATA/isles24_data/derivatives/',
          'dictFile': '/media/kimberly/DATA/isles24_data/partition_2D.pickle',
          'csvPath': '/media/kimberly/DATA/isles24_data/phenotype/',
          'resultsPath': '/media/kimberly/DATA/isles24_data/results/',
          'dim': (416, 416),
          'batch_size': 1,
          'timepoints': 32,
          'n_classes': 2,
          'features': [continuous_features, categorical_features['names']]}
