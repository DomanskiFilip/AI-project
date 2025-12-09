# Handwritten Digit Recognition System

A comprehensive machine learning system implementing and comparing multiple classification algorithms for handwritten digit recognition. This project explores traditional ML approaches including distance-based methods, neural networks, and ensemble techniques.

![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## Overview

This system implements **7 different classification algorithms** to recognize handwritten digits from 8×8 pixel matrices. Each digit is represented as a 64-pixel bitmap with grayscale values, providing a challenging yet manageable dataset for algorithm comparison.

### Key Features

- **Multiple Algorithms**: Euclidean Distance, K-Nearest Neighbors, Multi-Layer Perceptron, Support Vector Machines, Mahalanobis Distance, and more
- **Feature Engineering**: Genetic Algorithm optimization, K-Means clustering, and centroid-based features
- **Ensemble Learning**: "All at Once" voting system combining all algorithms
- **Two-Fold Cross-Validation**: Comprehensive testing on both dataset combinations
- **Modular Architecture**: Clean, professional code structure with 21+ files across 6 packages

## Project Structure

```
src/
├── main/
│   └── CWmain.java                    # Entry point
├── constants/
│   └── Constants.java                 # Global configuration
├── algorithms/
│   ├── Algorithm.java                 # Base interface
│   ├── EuclideanDistance.java         # Simple distance-based classification
│   ├── KNearestNeighbour.java         # K-NN implementation (K=3)
│   ├── DistanceFromCentroid.java      # Centroid-based classification
│   ├── MahalanobisDistance.java       # Statistical distance method
│   ├── MultiLayerPerceptron.java      # Neural network (300 perceptrons)
│   ├── SupportVectorMachine.java      # SVM with One-vs-Rest & One-vs-One
│   ├── AllAtOnce.java                 # Ensemble voting system
│   └── ...                            # Supporting classes
├── features/
│   ├── FeatureMode.java               # Feature selection strategies
│   └── FeatureExtractor.java          # Feature engineering utilities
├── evaluation/
│   ├── EvaluationResult.java          # Results data structure
│   └── Evaluator.java                 # Performance evaluation
├── data/
│   ├── DatasetReader.java             # CSV data loading
│   └── DatasetPrinter.java            # Dataset visualization
└── ui/
    └── UserInterface.java              # Interactive console UI
```

## Algorithm Performance

### Dataset Information
- **Size**: 2,810 samples per dataset
- **Features**: 64 pixels (8×8 grayscale matrix)
- **Classes**: 10 digits (0-9)
- **Format**: CSV with 65 columns (64 pixels + 1 label)

### Results Summary

| Algorithm | Avg Success Rate | Avg Time (s) | Key Characteristics |
|-----------|-----------------|--------------|---------------------|
| **Euclidean Distance** | **98.26%** | 0.53 | ⭐ Best overall performance |
| K-Nearest Neighbor | **98.11%** | 0.54 | Robust, considers local context |
| Mahalanobis Distance | 96.62% | 36.50 | Accounts for feature correlation |
| MLP [Raw Only] | 97.72% | ~5.00 | Neural network baseline |
| SVM [All Features] | 96.92% | 0.02 | Fast inference, good accuracy |
| Distance from Centroid | 90.43% | 0.42 | Simple but effective |
| All at Once (Ensemble) | **97.51%** | 35.35 | Combines all algorithms |

> **Observation**: Dataset B consistently produces better results when used for training, suggesting it contains more varied drawing styles and characteristics that improve generalization.

## Getting Started

### Prerequisites

- Java 11 or higher
- Datasets in CSV format (place in `datasets/` directory)
  - `dataSetA.csv`
  - `dataSetB.csv`

### Running the Project

#### Option 1: Run All Algorithms (Recommended)
```java
// In CWmain.java
public static void main(String[] args) {
    List<List<Integer>> dataSetA = DatasetReader.readCsvFile(DATASET_A_FILE_PATH);
    List<List<Integer>> dataSetB = DatasetReader.readCsvFile(DATASET_B_FILE_PATH);
    
    Evaluator.runAllInOrder(dataSetA, dataSetB);
}
```

This will:
- Run all algorithms on both dataset combinations (A→B and B→A)
- Display individual results for each algorithm
- Calculate and display average performance metrics

#### Option 2: Interactive UI
```java
// In CWmain.java
public static void main(String[] args) {
    List<List<Integer>> dataSetA = DatasetReader.readCsvFile(DATASET_A_FILE_PATH);
    List<List<Integer>> dataSetB = DatasetReader.readCsvFile(DATASET_B_FILE_PATH);
    
    UserInterface.start(dataSetA, dataSetB);
}
```

Interactive menu allows you to:
- Run individual algorithms
- Print dataset samples
- Compare specific algorithm variants
- Test different hyperparameters

### Compilation & Execution

```bash
# Compile
javac -d bin src/**/*.java

# Run
java -cp bin main.CWmain
```

## Algorithm Details

### 1. Euclidean Distance
**Best Performer** - Finds the nearest training sample to the test sample using L2 distance.

```
Success Rate: 98.26% | Time: 0.53s
```

- Simple yet highly effective for this task
- No training phase required
- Computationally intensive for large datasets

### 2. K-Nearest Neighbors (K=3)
Considers the 3 nearest neighbors and uses majority voting.

```
Success Rate: 98.11% | Time: 0.54s
```

- More robust than single nearest neighbor
- K=3 found to be optimal through experimentation
- Balances accuracy and computational cost

### 3. Multi-Layer Perceptron
Neural network with 300 hidden perceptrons, 50 epochs, learning rate 0.1.

```
Success Rate: 97.72% | Time: ~5s
```

**Feature Engineering Variants:**
- **Raw Only**: Original 64 pixels
- **Centroid Only**: Distances to class centroids
- **Raw + Centroid**: Combined features
- **Raw + K-Means**: With K-Means cluster distances
- **Raw + GA**: Genetic Algorithm weighted features
- **All Features**: Complete feature set

**Key Findings:**
- Raw pixels alone often outperform engineered features
- 500 perceptrons × 500 epochs best for A→B
- 1000 perceptrons × 50 epochs best for B→A

### 4. Support Vector Machine
Linear SVM with both One-vs-Rest and One-vs-One strategies.

```
One-vs-Rest: 96.92% | One-vs-One: 96.92% | Time: 0.02s
```

- Extremely fast inference
- Multiple feature modes supported
- Averaged perceptron for stability

### 5. Mahalanobis Distance
Statistical distance method accounting for feature correlations.

```
Success Rate: 96.62% | Time: 36.50s
```

- Per-class covariance matrices
- Better theoretical foundation than Euclidean
- Computationally expensive
- May require more samples for optimal performance

### 6. Distance from Centroid
Classifies based on nearest class centroid (center of mass).

```
Success Rate: 90.43% | Time: 0.42s
```

- Fast and simple
- Good baseline algorithm
- Works well when classes are well-separated

### 7. All at Once (Ensemble)
Runs all algorithms and selects the most voted digit.

```
Success Rate: 97.51% | Time: 35.35s
```

- Combines strengths of all algorithms
- Voting reveals interesting patterns:
  - Misclassifications often cluster around 2-3 candidates
  - Similar wrong predictions across different algorithms suggest specific challenging samples
  - Rarely more than 3 competing predictions

## Basic Feature Engineering Observations

### Genetic Algorithm (GA)
- **Population**: 20 individuals
- **Generations**: 100 (with early stopping at 99% fitness)
- **Mutation Rate**: 5%
- **Purpose**: Weight optimization for pixel importance

**Impact**: 10× population increase improved MLP by 0.2% but increased time 30×

### K-Means Clustering
- **Clusters**: 10
- **Max Iterations**: 20
- **Initialization**: K-Means++
- **Purpose**: Generate distance-based features

**Impact**: 10× iterations improved MLP by 0.1%

### Centroid Features
- Computes center of mass for each digit class
- Provides 10 distance features (one per class)
- Fast and effective for classification

## Hyperparameter Tuning

### MLP Optimal Settings

| Train/Test | Learning Rate | Epochs | Perceptrons | Success Rate |
|------------|---------------|--------|-------------|--------------|
| A → B | 0.1 | 500 | 500 | 97.72% |
| B → A | 0.1 | 50 | 1000 | 98.40% |

**Key Insights:**
- Higher learning rates (>0.1) degrade performance
- More perceptrons help with complex training sets (B)
- More epochs help with less varied training sets (A)

## Future Enhancements

- I am planning on makeing a web-based GUI for easier interaction.

## License

This project is available for educational and research purposes.

## Author

**Filip Domanski** - Middlesex Univeristy, Year 3 AI Course

## Interesting Findings

### Dataset B Superiority
When training on Dataset B and testing on A, success rates are consistently **0.4-0.8% higher** across all algorithms. This suggests Dataset B may contain:
- More varied handwriting styles
- More exaggerated digit features
- Better representation of edge cases
- Superior generalization characteristics

### Ensemble Behavior
Analyzing the "All at Once" voting patterns reveals:
- Misclassifications rarely have >3 competing predictions
- Most errors show 2 strong candidates (e.g., confusion between "2" and "8")
- Suggests systematic feature ambiguities rather than random errors
- Distance-based and neural methods often agree on misclassifications

### Feature Engineering Paradox
Despite sophisticated feature engineering:
- Raw pixels often outperform engineered features
- Simple algorithms (Euclidean) beat complex ones (MLP with features)
- Suggests the 8×8 representation already captures essential information
- Over-engineering may introduce noise or lose critical spatial relationships

---

⭐ **Star this repository if you find it interesting!**

**Questions?** Open an issue or reach out