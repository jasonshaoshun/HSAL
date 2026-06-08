import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load data from npz file"""
    data = np.load(filepath)
    return data

def train_mlp_with_dev(x_train, y_train, x_dev, y_dev, x_test, y_test, task_name):
    """Train MLP classifier with development set for early stopping"""
    random.seed(0)
    np.random.seed(0)
    
    best_dev_score = 0
    best_model = None
    patience = 10
    no_improve_count = 0
    
    # Progressive training with early stopping based on dev set
    for max_iter in [50, 100, 200, 300, 500, 800, 1000]:
        clf = MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            max_iter=max_iter,
            random_state=0
        )
        
        clf.fit(x_train, y_train)
        dev_score = clf.score(x_dev, y_dev)
        
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_model = clf
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= 3:  # Stop if no improvement for 3 iterations
            break
    
    clf = best_model
    
    # Calculate scores
    score_train = clf.score(x_train, y_train)
    score_dev = clf.score(x_dev, y_dev)
    score_test = clf.score(x_test, y_test)
    
    print(f"    {task_name} - Train: {score_train:.4f}, Dev: {score_dev:.4f}, Test: {score_test:.4f}")
    
    return clf, score_train, score_dev, score_test

def evaluate_model_dataset(model_path, model_name):
    """Evaluate MLP classifier on both reputation and location tasks"""
    try:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {model_name}")
        print(f"{'='*60}")
        
        # Load data
        data = load_data(model_path)
        print("Available keys:", list(data.keys()))
        
        # Extract data with correct key names
        x_train = data['x_train_aboutme_cls']
        x_dev = data['x_dev_aboutme_cls']
        x_test = data['x_test_aboutme_cls']
        
        y_train_reputation = data['y_train_reputation']
        y_dev_reputation = data['y_dev_reputation']
        y_test_reputation = data['y_test_reputation']
        
        y_train_location = data['y_train_location']
        y_dev_location = data['y_dev_location']
        y_test_location = data['y_test_location']
        
        print(f"Training data shape: X={x_train.shape}")
        print(f"Dev data shape: X={x_dev.shape}")
        print(f"Test data shape: X={x_test.shape}")
        print(f"Reputation classes: {len(np.unique(y_train_reputation))}")
        print(f"Location classes: {len(np.unique(y_train_location))}")
        
        results = {'model': model_name}
        
        # Train on reputation task
        print(f"\nTraining MLP for REPUTATION classification:")
        clf_rep, train_rep, dev_rep, test_rep = train_mlp_with_dev(
            x_train, y_train_reputation, x_dev, y_dev_reputation, x_test, y_test_reputation, "Reputation"
        )
        
        results['reputation'] = {
            'classifier': clf_rep,
            'train_score': train_rep,
            'dev_score': dev_rep,
            'test_score': test_rep,
            'overfitting': train_rep - test_rep
        }
        
        # Train on location task
        print(f"\nTraining MLP for LOCATION classification:")
        clf_loc, train_loc, dev_loc, test_loc = train_mlp_with_dev(
            x_train, y_train_location, x_dev, y_dev_location, x_test, y_test_location, "Location"
        )
        
        results['location'] = {
            'classifier': clf_loc,
            'train_score': train_loc,
            'dev_score': dev_loc,
            'test_score': test_loc,
            'overfitting': train_loc - test_loc
        }
        
        # Detailed evaluation
        print(f"\nDetailed Classification Reports:")
        print(f"\n--- REPUTATION TASK ---")
        y_pred_rep = clf_rep.predict(x_test)
        print(classification_report(y_test_reputation, y_pred_rep))
        
        print(f"\n--- LOCATION TASK ---")
        y_pred_loc = clf_loc.predict(x_test)
        print(classification_report(y_test_location, y_pred_loc))
        
        return results
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {model_path}")
        return None
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")
        return None

def main():
    # Define all model datasets
    models = {
        "BERT-base-uncased": "data/mSEFair-textual/bert-base-uncased/en.npz",
        "Llama-3.1-8B": "data/mSEFair-textual/meta-llama/Llama-3.1-8B/en.npz",
        "Llama-3.2-3B": "data/mSEFair-textual/meta-llama/Llama-3.2-3B/en.npz",
        "Meta-Llama-3-8B": "data/mSEFair-textual/meta-llama/Meta-Llama-3-8B/en.npz",
        "Mistral-7B-Instruct-v0.3": "data/mSEFair-textual/mistralai/Mistral-7B-Instruct-v0.3/en.npz",
        "Mistral-7B-v0.3": "data/mSEFair-textual/mistralai/Mistral-7B-v0.3/en.npz"
    }
    
    # models = {
    #     "BERT-base-uncased": "data/mSEFair-textual/bert-base-uncased/en.npz",
    # }

    results = []
    
    # Process each model dataset
    for model_name, model_path in models.items():
        result = evaluate_model_dataset(model_path, model_name)
        if result:
            results.append(result)
    
    # # Summary of all results
    # if results:
    #     print(f"\n{'='*90}")
    #     print("SUMMARY OF ALL MODELS - REPUTATION TASK")
    #     print(f"{'='*90}")
    #     print(f"{'Model':<25} {'Train':<8} {'Dev':<8} {'Test':<8} {'Overfit':<8}")
    #     print("-" * 65)
        
    #     for result in results:
    #         rep = result['reputation']
    #         print(f"{result['model']:<25} {rep['train_score']:.4f}   {rep['dev_score']:.4f}   {rep['test_score']:.4f}   {rep['overfitting']:.4f}")
        
    #     print(f"\n{'='*90}")
    #     print("SUMMARY OF ALL MODELS - LOCATION TASK")
    #     print(f"{'='*90}")
    #     print(f"{'Model':<25} {'Train':<8} {'Dev':<8} {'Test':<8} {'Overfit':<8}")
    #     print("-" * 65)
        
    #     for result in results:
    #         loc = result['location']
    #         print(f"{result['model']:<25} {loc['train_score']:.4f}   {loc['dev_score']:.4f}   {loc['test_score']:.4f}   {loc['overfitting']:.4f}")
        
    #     # Find best performing models for each task
    #     best_rep = max(results, key=lambda x: x['reputation']['test_score'])
    #     best_loc = max(results, key=lambda x: x['location']['test_score'])
        
    #     print(f"\nBest model for REPUTATION: {best_rep['model']} (Test: {best_rep['reputation']['test_score']:.4f})")
    #     print(f"Best model for LOCATION: {best_loc['model']} (Test: {best_loc['location']['test_score']:.4f})")
        
    # Summary of all results
    if results:
        print(f"\n{'='*90}")
        print("SUMMARY OF ALL MODELS - REPUTATION TASK")
        print(f"{'='*90}")
        print(f"{'Model':<25} {'Train':<8} {'Dev':<8} {'Test':<8}")
        print("-" * 65)
        
        for result in results:
            rep = result['reputation']
            print(f"{result['model']:<25} {rep['train_score']:.4f}   {rep['dev_score']:.4f}   {rep['test_score']:.4f}")
        
        print(f"\n{'='*90}")
        print("SUMMARY OF ALL MODELS - LOCATION TASK")
        print(f"{'='*90}")
        print(f"{'Model':<25} {'Train':<8} {'Dev':<8} {'Test':<8}")
        print("-" * 65)
        
        for result in results:
            loc = result['location']
            print(f"{result['model']:<25} {loc['train_score']:.4f}   {loc['dev_score']:.4f}   {loc['test_score']:.4f}")
        
        # Find best performing models for each task
        best_rep = max(results, key=lambda x: x['reputation']['test_score'])
        best_loc = max(results, key=lambda x: x['location']['test_score'])
        
        print(f"\nBest model for REPUTATION: {best_rep['model']} (Test: {best_rep['reputation']['test_score']:.4f})")
        print(f"Best model for LOCATION: {best_loc['model']} (Test: {best_loc['location']['test_score']:.4f})")
    
    return results

if __name__ == "__main__":
    results = main()