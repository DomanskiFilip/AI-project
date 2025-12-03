package main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Scanner;

public class CWmain {

    private static final String DATASET_A_FILE_PATH = "datasets/dataSetA.csv";
    private static final String DATASET_B_FILE_PATH = "datasets/dataSetB.csv";
    private static final int ENTIRE_BITMAP_SIZE = 65; // 64 bits + 1 category/class
    private static final int BITMAP_SIZE = ENTIRE_BITMAP_SIZE - 1; // Bitmap size minus bit representing digit
    private static final int BITMAPS_TO_DISPLAY = 20;


    // placeholder Algorythm interface
    public interface Algorithm {
        // sample -> row to predict, trainingSet -> dataset A (usually)
        Object predict(List<Integer> sample, List<List<Integer>> trainingSet); 
    }

    // Initialize non-configurable algorithms as constants
    private static final Algorithm EUCLIDEAN_DISTANCE = new EuclideanDistance();
    private static final Algorithm MULTI_LAYER_PERCEPTRON = new MultiLayerPerceptron();
    private static final Algorithm DISTANCE_FROM_CENTROID = new DistanceFromCentroid();
    // The SVM is instantiated dynamically in the UI to test different kernel functions
    private static final Algorithm K_NEAREST_NEIGHBOUR = new K_NEAREST_NEIGHBOUR();
    private static final Algorithm MAHALANOBIS_DISTANCE = new MahalanobisDistance();
    private static final Algorithm ALL_AT_ONCE = new AllAtOnce();


    // --- GLOBAL HELPER FUNCTIONS ---

    // function to calculate the centroid for each of the 10 classes
    public static double[][] calculateCentroids(List<List<Integer>> trainingSet) {
        double[][] centroids = new double[10][BITMAP_SIZE];
        double[][] sumPerClass = new double[10][BITMAP_SIZE];
        int[] countPerClass = new int[10];

        // Sum up pixel values for each class
        for (List<Integer> row : trainingSet) {
            int instance = row.get(BITMAP_SIZE); // The class label (0-9)
            countPerClass[instance]++;
            for (int i = 0; i < BITMAP_SIZE; i++) {
                sumPerClass[instance][i] += row.get(i);
            }
        }

        // Calculate the average (centroid)
        for (int digit = 0; digit < 10; digit++) {
            if (countPerClass[digit] > 0) {
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    centroids[digit][i] = sumPerClass[digit][i] / countPerClass[digit];
                }
            }
        }
        return centroids;
    }

    private static final int KMEANS_CLUSTERS = 10;
    private static final int KMEANS_MAX_ITERATIONS = 20;

    // function to compute K-Means centroids using K-Means++ initialization
    public static double[][] computeKMeansCentroids(List<List<Integer>> trainingSet, int clusters) {
        if (trainingSet == null || trainingSet.isEmpty()) {
            return new double[clusters][BITMAP_SIZE];
        }
        Random random = new Random(42);
        double[][] centroids = new double[clusters][BITMAP_SIZE];
        
        // K-Means++ Initialization: selects initial centroids intelligently
        List<Integer> first = trainingSet.get(random.nextInt(trainingSet.size()));
        for (int j = 0; j < BITMAP_SIZE; j++) {
            centroids[0][j] = first.get(j);
        }

        for (int k = 1; k < clusters; k++) {
            double[] dist = new double[trainingSet.size()];
            double total = 0;
            // Calculate distance to the nearest existing centroid for all samples
            for (int i = 0; i < trainingSet.size(); i++) {
                List<Integer> sample = trainingSet.get(i);
                double minDistance = Double.MAX_VALUE;
                for (int c = 0; c < k; c++) {
                    double distance = 0;
                    for (int f = 0; f < BITMAP_SIZE; f++) {
                        double diff = sample.get(f) - centroids[c][f];
                        distance += diff * diff;
                    }
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
                dist[i] = minDistance;
                total += minDistance; // Accumulate total distance squared
            }
            // Select new centroid weighted by distance
            double randomValue = random.nextDouble() * total;
            double cumulativeDistance = 0;
            for (int i = 0; i < dist.length; i++) {
                cumulativeDistance += dist[i];
                if (cumulativeDistance >= randomValue) {
                    List<Integer> picked = trainingSet.get(i);
                    for (int j = 0; j < BITMAP_SIZE; j++) {
                        centroids[k][j] = picked.get(j);
                    }
                    break;
                }
            }
        }

        // Iterative refinement (standard K-Means)
        for (int iter = 0; iter < KMEANS_MAX_ITERATIONS; iter++) {
            double[][] sums = new double[clusters][BITMAP_SIZE];
            int[] counts = new int[clusters];
            
            // Assign points to nearest centroid
            for (List<Integer> s : trainingSet) {
                int best = 0;
                double bestD = Double.MAX_VALUE;
                for (int c = 0; c < clusters; c++) {
                    double d = 0;
                    for (int f = 0; f < BITMAP_SIZE; f++) {
                        double diff = s.get(f) - centroids[c][f];
                        d += diff * diff;
                    }
                    if (d < bestD) {
                        bestD = d;
                        best = c;
                    }
                }
                counts[best]++;
                for (int f = 0; f < BITMAP_SIZE; f++) sums[best][f] += s.get(f);
            }
            
            // Recalculate centroids
            boolean moved = false;
            for (int c = 0; c < clusters; c++) {
                if (counts[c] > 0) {
                    for (int f = 0; f < BITMAP_SIZE; f++) {
                        double newVal = sums[c][f] / counts[c];
                        if (Math.abs(newVal - centroids[c][f]) > 1e-6) {
                            moved = true;
                        }
                        centroids[c][f] = newVal;
                    }
                }
            }
            if (!moved) {
                break; // Optimization: stop if convergence reached
            }
        }
        return centroids;
    }

    // function to compute Euclidean distances from a sample to the K-Means centroids
    public static double[] computeKMeansDistances(List<Integer> sample, double[][] kmeansCentroids) {
        if (kmeansCentroids == null) {
            return new double[0];
        }
        int clusters = kmeansCentroids.length;
        double[] dists = new double[clusters];
        for (int c = 0; c < clusters; c++) {
            double sum = 0;
            for (int f = 0; f < BITMAP_SIZE; f++) {
                double diff = sample.get(f) - kmeansCentroids[c][f];
                sum += diff * diff;
            }
            dists[c] = Math.sqrt(sum);
        }
        return dists;
    }

    private static final int GA_POPULATION = 20;
    private static final int GA_GENERATIONS = 20;
    private static final double GA_MUTATION = 0.05;

    // function to evolve weights using a Genetic Algorithm for feature selection/emphasis
    public static double[] evolveGeneticWeights(List<List<Integer>> trainingSet) {
        Random randomGenerator = new Random(42);
        List<double[]> population = new ArrayList<>();
        // Initialize population (random weights)
        for (int i = 0; i < GA_POPULATION; i++) {
            double[] individual = new double[BITMAP_SIZE];
            for (int f = 0; f < BITMAP_SIZE; f++) individual[f] = randomGenerator.nextDouble();
            population.add(individual);
        }

        double fitnessThreshold = 0.99; // Stop if fitness reaches this value
        int maxGenerations = GA_GENERATIONS * 5;
        double bestScore = -1;
        double[] bestIndividual = null;
        for (int generation = 0; generation < maxGenerations; generation++) {
            // Evaluate fitness for all individuals
            double[] fitnessScores = new double[population.size()];
            int bestIndex = 0;
            for (int i = 0; i < population.size(); i++) {
                fitnessScores[i] = evaluateWeightFitness(population.get(i), trainingSet);
                if (fitnessScores[i] > bestScore) {
                    bestScore = fitnessScores[i];
                    bestIndividual = population.get(i);
                    bestIndex = i;
                }
            }
            // Early stop if threshold reached
            if (bestScore >= fitnessThreshold) break;

            // keep the best, replace others with children
            List<double[]> nextGeneration = new ArrayList<>();
            nextGeneration.add(bestIndividual.clone()); // Keep best
            while (nextGeneration.size() < population.size()) {
                // Randomly select two parents (excluding best)
                int parent1Idx = randomGenerator.nextInt(population.size());
                int parent2Idx = randomGenerator.nextInt(population.size());
                if (parent1Idx == bestIndex) parent1Idx = (parent1Idx + 1) % population.size();
                if (parent2Idx == bestIndex) parent2Idx = (parent2Idx + 1) % population.size();
                double[] parent1 = population.get(parent1Idx);
                double[] parent2 = population.get(parent2Idx);
                // Crossover
                double[] childIndividual = new double[BITMAP_SIZE];
                for (int f = 0; f < BITMAP_SIZE; f++) childIndividual[f] = randomGenerator.nextDouble() < 0.5 ? parent1[f] : parent2[f];
                // Mutation
                for (int f = 0; f < BITMAP_SIZE; f++) if (randomGenerator.nextDouble() < GA_MUTATION) childIndividual[f] += randomGenerator.nextGaussian() * 0.1;
                nextGeneration.add(childIndividual);
            }
            population = nextGeneration;
        }
        return bestIndividual;
    }

    // Fitness function: accuracy of a centroid classifier when using the given weights
    private static double evaluateWeightFitness(double[] weights, List<List<Integer>> trainingSet) {
        double[][] centroids = calculateCentroids(trainingSet);
        int correctMatches = 0;
        for (List<Integer> sample : trainingSet) {
            int bestClass = 0; double bestDistance = Double.MAX_VALUE;
            // Classify sample using weighted distance to centroids
            for (int c = 0; c < centroids.length; c++) {
                double distance = 0;
                for (int f = 0; f < BITMAP_SIZE; f++) {
                    // Apply weight before calculating difference
                    double diff = (sample.get(f) * weights[f]) - centroids[c][f]; 
                    distance += diff * diff;
                }
                if (distance < bestDistance) { bestDistance = distance; bestClass = c; }
            }
            if (bestClass == sample.get(BITMAP_SIZE)) correctMatches++;
        }
        return correctMatches / (double) Math.max(1, trainingSet.size());
    }

    // Convert raw pixel values to a double vector
    public static double[] buildRawPixelsVector(List<Integer> sample) {
        double[] v = new double[BITMAP_SIZE];
        for (int i = 0; i < BITMAP_SIZE; i++) v[i] = sample.get(i);
        return v;
    }

    // Create a feature vector of distances to the 10 class centroids
    public static double[] buildCentroidDistanceVector(List<Integer> sample, double[][] centroids) {
        int n = (centroids != null) ? centroids.length : 0;
        double[] v = new double[n];
        for (int d = 0; d < n; d++) {
            double sum = 0;
            for (int f = 0; f < BITMAP_SIZE; f++) {
                double diff = sample.get(f) - centroids[d][f];
                sum += diff * diff;
            }
            v[d] = Math.sqrt(sum);
        }
        return v;
    }

    // Create a feature vector of distances to the K-Means cluster centers
    public static double[] buildKMeansDistanceVector(List<Integer> sample, double[][] kmeansCentroids) {
        if (kmeansCentroids == null) {
            return new double[0];
        }
        int clusters = kmeansCentroids.length;
        double[] v = new double[clusters];
        for (int c = 0; c < clusters; c++) {
            double sum = 0;
            for (int f = 0; f < BITMAP_SIZE; f++) {
                double diff = sample.get(f) - kmeansCentroids[c][f];
                sum += diff * diff;
            }
            v[c] = Math.sqrt(sum);
        }
        return v;
    }

    // Create a feature vector of raw pixels multiplied by the Genetic Algorithm weights
    public static double[] buildGAWeightedVector(List<Integer> sample, double[] gaWeights) {
        double[] v = new double[BITMAP_SIZE];
        for (int i = 0; i < BITMAP_SIZE; i++) v[i] = sample.get(i) * (gaWeights != null ? gaWeights[i] : 1.0);
        return v;
    }

    // Utility function to combine multiple feature vectors into one long vector
    public static double[] concatVectors(double[]... parts) {
        int total = 0;
        for (double[] p : parts) if (p != null) total += p.length;
        double[] out = new double[total];
        int pos = 0;
        for (double[] p : parts) {
            if (p == null) continue;
            System.arraycopy(p, 0, out, pos, p.length);
            pos += p.length;
        }
        return out;
    }
    
    // Combination of all feature vectors used in the 'ALL' mode of SVM
    private static double[] buildCombinedFeatureVector(List<Integer> sample, double[][] centroidCache, double[][] kmeansCache, double[] gaWeightsCache) {
        // Raw pixels
        double[] raw = buildRawPixelsVector(sample);
        // Distances to class centroids
        double[] cent = buildCentroidDistanceVector(sample, centroidCache);
        // Distances to K-Means centroids
        double[] km = buildKMeansDistanceVector(sample, kmeansCache);
        // GA-weighted raw pixels
        double[] ga = buildGAWeightedVector(sample, gaWeightsCache);
        
        // Concatenate all of them
        return concatVectors(raw, cent, km, ga);
    }

    // ------------------------------------------------------------------------
    // AI ALGORITHMS:
    // ------------------------------------------------------------------------

    // Euclidean Distance Algorithm
    private static class EuclideanDistance implements Algorithm {

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (trainingSet == null || trainingSet.isEmpty()) throw new IllegalArgumentException("Dataset empty");
            double minDistance = Double.MAX_VALUE;
            List<Integer> closest = null;
            for (List<Integer> candidate : trainingSet) {
                double sum = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double distance = sample.get(i) - candidate.get(i);
                    sum += distance * distance; // Squared Euclidean Distance gives better results then square root
                }
                if (sum < minDistance) {
                    minDistance = sum;
                    closest = candidate;
                }
            }
            return closest != null ? Integer.valueOf(closest.get(BITMAP_SIZE)) : Integer.valueOf(-1);
        }
    }

    // Multi Layer Perceptron Algorythm
    private static class MultiLayerPerceptron implements Algorithm {
        // modify these hyper parameters to tune the neural network:
        private static final int PERCEPTRONS = 100; // number of neurons in the hidden layer
        private static final int EPOCHS = 50; // number of training iterations
        // 500 perceptrons and 500 epochs fairs best for training on A and testing on B
        // 1000 perceptrons and 50 epochs fairs best for training on B and testing on A
        private static final double LEARNING_RATE = 0.1;
        private static final long RANDOM_SEED = 42; // Fixed seed for the random number generator to ensure reproducibility.

        // fixed parameters:
        private static final int CLASSES = 10; // number of classes (digits 0-9)
        private double[][] weightsInputHidden;  // weights between input and hidden layer [perceptron][features]
        private double[] biasHidden;
        private double[][] weightsHiddenOutput; // weights between hidden and output layer [classes][perceptron]
        private double[] biasOutput;
        private boolean trained = false;

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (!trained) {
                train(trainingSet);
            }
            double[] inputs = toInputVector(sample); // convert input sample to double array
            double[] outputs = forward(inputs); // forward propagation to get output activations
            int bestIndex = 0;
            double bestValue = outputs[0];
            for (int i = 1; i < outputs.length; i++) {
                if (outputs[i] > bestValue) {
                    bestValue = outputs[i];
                    bestIndex = i;
                }
            }
            return Integer.valueOf(bestIndex);
        }

        // method to train the multi layer perceptron (neural network) using backpropagation
        private void train(List<List<Integer>> trainingSet) {
            initializeWeights();

            // loop running epochs of training
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                for (List<Integer> row : trainingSet) {
                    double[] inputs = toInputVector(row);
                    int targetClass = row.get(BITMAP_SIZE);
                    double[] hidden = new double[PERCEPTRONS];
                    double[] outputs = new double[CLASSES];

                    // forward pass to calculate hidden layer
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        double sum = biasHidden[h];
                        for (int i = 0; i < BITMAP_SIZE; i++) {
                            sum += weightsInputHidden[h][i] * inputs[i];
                        }
                        hidden[h] = sigmoid(sum);
                    }

                    // forward pass to calculate output layer
                    for (int o = 0; o < CLASSES; o++) {
                        double sum = biasOutput[o];
                        for (int h = 0; h < PERCEPTRONS; h++) {
                            sum += weightsHiddenOutput[o][h] * hidden[h];
                        }
                        outputs[o] = sigmoid(sum);
                    }

                    double[] target = new double[CLASSES];
                    target[targetClass] = 1.0;

                    double[] outputDeltas = new double[CLASSES];
                    // calculate gradients for output layer
                    for (int o = 0; o < CLASSES; o++) {
                        double error = target[o] - outputs[o];
                        outputDeltas[o] = error * sigmoidDerivative(outputs[o]);
                    }

                    double[] hiddenDeltas = new double[PERCEPTRONS];
                    // calculate gradients for hidden layer
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        double error = 0;
                        for (int o = 0; o < CLASSES; o++) {
                            error += outputDeltas[o] * weightsHiddenOutput[o][h];
                        }
                        hiddenDeltas[h] = error * sigmoidDerivative(hidden[h]);
                    }

                    // update weights and biases between hidden layer
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        for (int i = 0; i < BITMAP_SIZE; i++) {
                            weightsInputHidden[h][i] += LEARNING_RATE * hiddenDeltas[h] * inputs[i];
                        }
                        biasHidden[h] += LEARNING_RATE * hiddenDeltas[h];
                    }

                    // update weights and biases between output layer
                    for (int o = 0; o < CLASSES; o++) {
                        for (int h = 0; h < PERCEPTRONS; h++) {
                            weightsHiddenOutput[o][h] += LEARNING_RATE * outputDeltas[o] * hidden[h];
                        }
                        biasOutput[o] += LEARNING_RATE * outputDeltas[o];
                    } 
                }
            }

            trained = true;
        }

        // method to initialize weights and biases
        private void initializeWeights() {
            weightsInputHidden = new double[PERCEPTRONS][BITMAP_SIZE];
            biasHidden = new double[PERCEPTRONS];
            weightsHiddenOutput = new double[CLASSES][PERCEPTRONS];
            biasOutput = new double[CLASSES];
            Random random = new Random(RANDOM_SEED);

            double range = 0.1; // maximum absolute value for initial weights and biases
            for (int h = 0; h < PERCEPTRONS; h++) {
                biasHidden[h] = (random.nextDouble() * 2 - 1) * range;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    weightsInputHidden[h][i] = (random.nextDouble() * 2 - 1) * range;
                }
            }
            for (int o = 0; o < CLASSES; o++) {
                biasOutput[o] = (random.nextDouble() * 2 - 1) * range;
                for (int h = 0; h < PERCEPTRONS; h++) {
                    weightsHiddenOutput[o][h] = (random.nextDouble() * 2 - 1) * range;
                }
            }
        }

        // method for forward propagation (useing trained weights to predict the class)
        private double[] forward(double[] inputs) {
            double[] hidden = new double[PERCEPTRONS];
            for (int h = 0; h < PERCEPTRONS; h++) {
                double sum = biasHidden[h];
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    sum += weightsInputHidden[h][i] * inputs[i];
                }
                hidden[h] = sigmoid(sum);
            }

            double[] outputs = new double[CLASSES];
            for (int o = 0; o < CLASSES; o++) {
                double sum = biasOutput[o];
                for (int h = 0; h < PERCEPTRONS; h++) {
                    sum += weightsHiddenOutput[o][h] * hidden[h];
                }
                outputs[o] = sigmoid(sum);
            }
            return outputs;
        }
        
        // method to convert inputed sample into input vector (from int -> double)
        private double[] toInputVector(List<Integer> sample) {
            double[] inputs = new double[BITMAP_SIZE];
            for (int i = 0; i < BITMAP_SIZE; i++) {
                inputs[i] = sample.get(i);
            }
            return inputs;
        }

        // Sigmoid method to describe sides of the perceptron activation
        private double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x)); // exp() computes how the sigmoid will grow and divide the space
        }

        // Derivative of the sigmoid method for backpropagation to adjust weights converting error to gradient
        private double sigmoidDerivative(double activatedValue) {
            return activatedValue * (1.0 - activatedValue);
        }

        // Expose the parameters used for calculation
        public String getParameters() {
            return "Epochs: " + EPOCHS + ", Learning rate: " + LEARNING_RATE  + ", perceptrons: " + PERCEPTRONS + ", Random seed: " + RANDOM_SEED;
        }
    }

    // Distance From Centroid Algorithm
    private static class DistanceFromCentroid implements Algorithm {
        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            // Compute the center of mass for each digit class
            double[][] centroids = calculateCentroids(trainingSet);
            double minDistance = Double.MAX_VALUE;
            int closestClass = -1;
            
            // Compare the sample to each centroid
            for (int digit = 0; digit < 10; digit++) {
                double sum = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double diff = sample.get(i) - centroids[digit][i];
                    sum += diff * diff; // Squared Euclidean Distance gives better results then square root
                }
                double distance = Math.sqrt(sum);
                if (distance < minDistance) { 
                    minDistance = distance; 
                    closestClass = digit; 
                }
            }
            return Integer.valueOf(closestClass);
        }
    }

    // Support Vector Machine Algorithm
    private static class SupportVectorMachine implements Algorithm {
        
        // Enum to control which features/kernels to use for the SVM training
        public enum FeatureMode {
            ALL,            // Raw + Centroid + KMeans + GA (most complex kernel)
            CENTROID_ONLY,  // Only Distances to Class Centroids (often a strong baseline)
            RAW_CENTROID,   // Raw Pixels + Centroid Distances
            RAW_KMEANS,     // Raw Pixels + K-Means Distances
            RAW_GA          // Raw Pixels + GA Weighted Pixels
        }

        private static final int CLASSES = 10;
        private static final int MAX_EPOCHS = 10;
        private static final double LEARNING_RATE = 0.02;
        private static final double MARGIN = 0.002; // Soft-margin
        private static final long RANDOM_SEED = 42;

        private final FeatureMode mode;
        private boolean trained = false;

        // Pre-calculated caches to avoid re-computing for every prediction
        private double[][] centroidCache;
        private double[][] kmeansCentroidCache;
        private double[] gaWeightsCache;

        // The actual learned weights and biases
        private Model oneVsRestModel; // One-vs-Rest model (10 binary classifiers)
        private List<Model> oneVsOneModels; // One-vs-One models (45 binary classifiers)

        public SupportVectorMachine(FeatureMode mode) {
            this.mode = mode;
        }

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (!trained) {
                train(trainingSet);
                trained = true;
            }
            
            // Project input sample into the feature space defined by the current mode
            double[] rawFeatures = projectToFeatureSpace(sample);
            
            // Normalize features using the statistics learned during training
            double[] features = oneVsRestModel.normalizer.normalize(rawFeatures);

            // Predict using both classification schemes
            int oneVsRestPrediction = oneVsRestModel.predictOneVsRest(features);
            int oneVsOnePrediction = classifyOneVsOne(features);

            // Return both predictions
            return new int[]{oneVsRestPrediction, oneVsOnePrediction};
        }
        
        public boolean isTrained() {
             return trained; 
        }

        // calculates caches, normalizes, and trains One-vs-Rest and One-vs-One models
        private void train(List<List<Integer>> trainingSet) {
            // Determine which caches are needed based on the current FeatureMode
            boolean needCentroids = (mode == FeatureMode.ALL || mode == FeatureMode.CENTROID_ONLY || mode == FeatureMode.RAW_CENTROID);
            boolean needKMeans = (mode == FeatureMode.ALL || mode == FeatureMode.RAW_KMEANS);
            boolean needGA = (mode == FeatureMode.ALL || mode == FeatureMode.RAW_GA);

            // Compute the caches
            if(needCentroids) centroidCache = calculateCentroids(trainingSet);
            if(needKMeans) kmeansCentroidCache = computeKMeansCentroids(trainingSet, KMEANS_CLUSTERS);
            if(needGA) gaWeightsCache = evolveGeneticWeights(trainingSet);

            int n = trainingSet.size();
            // Projects all samples to the feature space to build the training matrix
            double[] firstSample = projectToFeatureSpace(trainingSet.get(0));
            int featureSize = firstSample.length;

            double[][] featureMatrix = new double[n][featureSize];
            int[] labels = new int[n];

            for (int i = 0; i < n; i++) {
                List<Integer> row = trainingSet.get(i);
                featureMatrix[i] = projectToFeatureSpace(row);
                labels[i] = row.get(BITMAP_SIZE);
            }

            // calculate mean/standard deviation and scale the training data for normalization purpouses 
            // Xnormalized = (X - mean) / standard deviation
            FeatureNormalizer normalizer = new FeatureNormalizer(featureSize);
            normalizer.fitAndTransform(featureMatrix); // Scales the training data in place

            // Train One-vs-Rest (10 classifiers vs all others)
            this.oneVsRestModel = trainPerceptronModel(featureMatrix, labels, normalizer);

            // Train One-vs-One (45 binary classifiers: 0 vs 1, 0 vs 2, ..., 8 vs 9)
            this.oneVsOneModels = new ArrayList<>();
            for (int i = 0; i < CLASSES; i++) {
                for (int j = i + 1; j < CLASSES; j++) {
                    List<Integer> pairIndices = new ArrayList<>();
                    for (int k = 0; k < n; k++){
                        if (labels[k] == i || labels[k] == j) {
                            pairIndices.add(k);
                        }
                    } 
                    
                    if(!pairIndices.isEmpty()) {
                        // Train a binary classifier for this specific pair
                        Model pairModel = trainLinearPerceptron(featureMatrix, labels, pairIndices, i, j);
                        oneVsOneModels.add(pairModel);
                    }
                }
            }
        }

        // maps the input sample to the selected bitmap / feature vector size
        private double[] projectToFeatureSpace(List<Integer> sample) {
            double[] raw = buildRawPixelsVector(sample);
            
            switch (mode) {
                case CENTROID_ONLY:
                    return buildCentroidDistanceVector(sample, centroidCache);
                case RAW_CENTROID:
                    return concatVectors(raw, buildCentroidDistanceVector(sample, centroidCache));
                case RAW_KMEANS:
                    return concatVectors(raw, buildKMeansDistanceVector(sample, kmeansCentroidCache));
                case RAW_GA:
                    return concatVectors(raw, buildGAWeightedVector(sample, gaWeightsCache));
                case ALL:
                default:
                    return buildCombinedFeatureVector(sample, centroidCache, kmeansCentroidCache, gaWeightsCache);
            }
        }

        // Aggregates votes from all 45 One-vs-One classifiers
        private int classifyOneVsOne(double[] features) {
            int[] votes = new int[CLASSES];
            for (Model model : oneVsOneModels) {
                // Score >= 0 votes for positive class, Score < 0 votes for negative class
                double score = model.pointInFeatureSpace(0, features) + model.bias[0];
                if (score >= 0) {
                    votes[model.positiveClassLabel]++;
                } else {
                    votes[model.negativeClassLabel]++;
                }
            }
            // Select the class with the maximum number of votes
            int bestClass = 0;
            int maxVotes = -1;
            for (int c = 0; c < CLASSES; c++) {
                if (votes[c] > maxVotes) {
                    maxVotes = votes[c];
                    bestClass = c;
                }
            }
            return bestClass;
        }

        // Training for the Multiclass One-vs-Rest model
        private Model trainPerceptronModel(double[][] features, int[] labels, FeatureNormalizer normalizer) {
            int featureSize = features[0].length;
            Model model = new Model(CLASSES, featureSize, normalizer);
            int n = features.length;
            List<Integer> indices = new ArrayList<>();
            for(int i = 0; i < n; i++) {
                indices.add(i);
            }
            Random rand = new Random(RANDOM_SEED);

            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                Collections.shuffle(indices, rand); // Shuffle data for stochastic training (suffling the dataset to avoid bias)
                int mistakes = 0;
                for (int idx : indices) {
                    double[] sample = features[idx];
                    int target = labels[idx];
                    PredictionResult res = model.computeScores(sample);
                    
                    // Find the best scoring rival class (the one predicted incorrectly)
                    int bestRival = -1;
                    double bestRivalScore = Double.NEGATIVE_INFINITY;
                    for (int c = 0; c < CLASSES; c++) {
                        if (c == target) {
                            continue;
                        }
                        if (res.scores[c] > bestRivalScore) { 
                            bestRivalScore = res.scores[c]; bestRival = c; 
                        }
                    }

                    // Check for margin violation (target score too low or rival score too high)
                    boolean violation = (res.scores[target] - bestRivalScore) <= MARGIN;
                    if (violation) {
                        mistakes++;
                        // increase weight for target class, decrease weight for rival class
                        model.update(sample, target, 1.0);
                        if (bestRival >= 0) {
                            model.update(sample, bestRival, -1.0);
                        }
                    }
                    model.accumulateAverages(); // Accumulate for the final averaged weights
                }
                if (mistakes == 0) {
                    break;
                }
            }
            model.finalizeWeights(); // Use the average of all weights/biases computed during training
            return model;
        }

        // Training for a binary / linear (One-vs-One) model
        private Model trainLinearPerceptron(double[][] features, int[] labels, List<Integer> indices, int classA, int classB) {
            int featureSize = features[0].length;
            Model model = new Model(1, featureSize, null); // Only one classifier (index 0)
            model.positiveClassLabel = classA; // Class A is the positive label (y=1)
            model.negativeClassLabel = classB; // Class B is the negative label (y=-1)
            Random rand = new Random(RANDOM_SEED + classA * 100 + classB);
            
            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                Collections.shuffle(indices, rand);
                boolean updated = false;
                for (int idx : indices) {
                    // Target label is +1 for classA, -1 for classB
                    int y = (labels[idx] == classA) ? 1 : -1;
                    double score = model.pointInFeatureSpace(0, features[idx]) + model.bias[0];
                    
                    // Perceptron learning rule
                    if (y * score <= MARGIN) {
                        for (int f = 0; f < featureSize; f++) {
                            model.weights[0][f] += LEARNING_RATE * y * features[idx][f];
                        }
                        model.bias[0] += LEARNING_RATE * y;
                        updated = true;
                    }
                    model.accumulateAverages();
                }
                if (!updated) break;
            }
            model.finalizeWeights();
            return model;
        }

        // Inner class: Holds the weights and logic for a single or multi-class perceptron model
        private static class Model {
            double[][] weights; 
            double[] bias;
            double[][] weightSums; // For averaging
            double[] biasSums; // For averaging
            long steps = 0;
            final FeatureNormalizer normalizer;
            int positiveClassLabel = -1; // Used only for One-vs-One
            int negativeClassLabel = -1; // Used only for One-vs-One

            Model(int numClasses, int numFeatures, FeatureNormalizer norm) {
                this.weights = new double[numClasses][numFeatures];
                this.bias = new double[numClasses];
                this.weightSums = new double[numClasses][numFeatures];
                this.biasSums = new double[numClasses];
                this.normalizer = norm;
            }

            // Computes the score for each class and identifies the prediction
            PredictionResult computeScores(double[] features) {
                double[] scores = new double[weights.length];
                int predicted = 0;
                double max = Double.NEGATIVE_INFINITY;
                for(int c = 0; c < weights.length; c++) {
                    scores[c] = pointInFeatureSpace(c, features) + bias[c];
                    if(scores[c] > max) {
                         max = scores[c]; predicted = c; 
                    }
                }
                return new PredictionResult(predicted, scores);
            }

            // Simple wrapper for One-vs-Rest prediction
            int predictOneVsRest(double[] features) { 
                return computeScores(features).predictedClass; 
            }

            // Calculates the point in feature space: feature vector * weight vector
            double pointInFeatureSpace(int classIdx, double[] features) {
                double sum = 0;

                for(int i = 0; i < features.length; i++) {
                    sum += weights[classIdx][i] * features[i];
                } 

                return sum;
            }

            // Applies the learning update to the weights and bias for a given class
            void update(double[] features, int classIdx, double direction) {
                for(int i = 0; i < features.length; i++) {
                    weights[classIdx][i] += LEARNING_RATE * direction * features[i];
                }
                bias[classIdx] += LEARNING_RATE * direction;
            }

            // Adds current weights/biases to the running sums for averaging
            void accumulateAverages() {
                steps++;
                for(int c = 0; c < weights.length; c++) {
                    for(int f = 0; f < weights[0].length; f++) {
                        weightSums[c][f] += weights[c][f];
                    } 
                    biasSums[c] += bias[c];
                }
            }

            // Calculates the final averaged weights
            void finalizeWeights() {
                if (steps == 0) {
                    return;
                }
                for(int c = 0; c<weights.length; c++) {
                    for(int f = 0; f < weights[0].length; f++) {
                        weights[c][f] = weightSums[c][f] / steps;
                    }
                    bias[c] = biasSums[c] / steps;
                }
            }
        }

        // Inner class: Standard scaling (Z-score normalization) for feature vectors (bitmaps) so that each feature has a mean of 0 and a standard deviation of 1.
        private static class FeatureNormalizer {
            double[] mean;
            double[] std;
            int featureSize;
            
            FeatureNormalizer(int size) {
                this.featureSize = size;
                this.mean = new double[size];
                this.std = new double[size];
            }
            
            // Calculates mean/standard deviation (fit) and applies normalization to the matrix (transform)
            void fitAndTransform(double[][] matrix) {
                if(matrix.length == 0) {
                    return;
                }
                
                // Calculates Mean and Standard Deviation (Fit)
                for (int f = 0; f < featureSize; f++) {
                    double sum = 0, sumSq = 0;
                    for (double[] row : matrix) {
                        sum += row[f];
                        sumSq += row[f] * row[f];
                    }
                    mean[f] = sum / matrix.length;
                    double var = (sumSq / matrix.length) - (mean[f] * mean[f]);
                    // Use a small epsilon to prevent division by zero for constant features
                    std[f] = Math.max(Math.sqrt(Math.max(var, 0)), 1e-9); 
                }
                
                // Apply Normalization (Transform)
                for (double[] row : matrix) {
                    for (int f = 0; f < featureSize; f++) {
                        row[f] = (row[f] - mean[f]) / std[f];
                    }
                }
            }
            
            // Applies normalization to a single input vector using pre-calculated stats
            double[] normalize(double[] vector) {
                double[] out = new double[featureSize];
                for (int f = 0; f < featureSize; f++) out[f] = (vector[f] - mean[f]) / std[f];
                return out;
            }
        }
        
        // Inner class: Simple data structure to hold results from computeScores
        private static class PredictionResult {
            int predictedClass;
            double[] scores;
            PredictionResult(int p, double[] s) {
                 this.predictedClass = p; this.scores = s; 
            }
        }
    }

    // K-Nearest Neighbour Algorithm
    private static class K_NEAREST_NEIGHBOUR implements Algorithm {
        private static final int K = 3;

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            PriorityQueue<double[]> heap = new PriorityQueue<>((a, b) -> Double.compare(b[0], a[0])); 
            
            for (List<Integer> candidate : trainingSet) {
                double distance = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double diff = sample.get(i) - candidate.get(i);
                    distance += diff * diff; // Squared Euclidean Distance gives better results than square root
                }
                
                // Keep only the K smallest distances
                if (heap.size() < K) {
                    heap.offer(new double[] {distance, candidate.get(BITMAP_SIZE)});
                } else if (distance < heap.peek()[0]) {
                    heap.poll();
                    heap.offer(new double[] {distance, candidate.get(BITMAP_SIZE)});
                }
            }
            
            // count votes for each class among the K nearest neighbors
            int[] votes = new int[10];
            while (!heap.isEmpty()) { 
                int digit = (int) heap.poll()[1]; 
                votes[digit]++; 
            }
            
            // Return the majority class
            int bestDigit = 0; int bestVotes = 0;
            for (int digit = 0; digit < votes.length; digit++) {
                if (votes[digit] > bestVotes) { 
                    bestVotes = votes[digit]; bestDigit = digit; 
                }
            }
            return Integer.valueOf(bestDigit);
        }
    }

    // Mahalanobis Distance Algorithm
    private static class MahalanobisDistance implements Algorithm {
        private static final int CLASSES = 10;
        
        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            int featureCount = BITMAP_SIZE; // Only using the 64 pixels as features
            
            // Pre-calculations needed for Mahalanobis Distance
            double[][] centroids = calculateCentroids(trainingSet);
            int[] classCounts = new int[CLASSES];
            for (List<Integer> row : trainingSet) {
                classCounts[row.get(BITMAP_SIZE)]++;
            }
            double[] featureMeans = computeFeatureMeans(trainingSet, featureCount);
            
            // Calculate the global covariance matrix
            double[][] covariance = computeCovarianceMatrix(trainingSet, featureCount, featureMeans);
            // Invert the matrix (this is the expensive step)
            double[][] inverseCovariance = invertMatrix(covariance); 
            if (inverseCovariance == null) {
                return Integer.valueOf(-1); // Safety check for singular matrix
            }

            double[] sampleVector = new double[featureCount];
            for (int i = 0; i < featureCount; i++) {
                sampleVector[i] = sample.get(i);
            } 

            // Calculate Mahalanobis distance to each class centroid
            double bestDistance = Double.MAX_VALUE;
            int bestClass = -1;
            for (int digit = 0; digit < CLASSES; digit++) {
                if (classCounts[digit] == 0) {
                    continue;
                }
                
                // Difference vector: Sample - Centroid
                double[] diff = new double[featureCount];
                for (int i = 0; i < featureCount; i++) {
                    diff[i] = sampleVector[i] - centroids[digit][i];
                }
                
                // Mahalanobis Distance: sqrt( diff^T * Covariance^-1 * diff )
                double distance = computeMahalanobisDistance(diff, inverseCovariance);
                if (distance < bestDistance) {
                    bestDistance = distance; bestClass = digit; 
                }
            }
            return Integer.valueOf(bestClass >= 0 ? bestClass : 0);
        }
        
        // Calculates the mean for each of the 64 features
        private static double[] computeFeatureMeans(List<List<Integer>> trainingSet, int featureCount) {
            double[] means = new double[featureCount];
            for (List<Integer> row : trainingSet) {
                for (int i = 0; i < featureCount; i++) {
                    means[i] += row.get(i);
                }
            }

            for (int i = 0; i < featureCount; i++) {
                means[i] /= trainingSet.size();
            }
            return means;
        }
        
        // Calculates the Covariance Matrix
        private static double[][] computeCovarianceMatrix(List<List<Integer>> trainingSet, int featureCount, double[] means) {
            double[][] covariance = new double[featureCount][featureCount];
            if (trainingSet.size() <= 1) {
                return covariance;
            }
            
            for (List<Integer> row : trainingSet) {
                for (int i = 0; i < featureCount; i++) {
                    double diffI = row.get(i) - means[i];
                    for (int j = 0; j < featureCount; j++) {
                        double diffJ = row.get(j) - means[j];
                        covariance[i][j] += diffI * diffJ; // Sum of outer products
                    }
                }
            }
            
            // Normalize and add regularization to the diagonal
            double denom = trainingSet.size() - 1.0;
            for (int i = 0; i < featureCount; i++) {
                for (int j = 0; j < featureCount; j++) {
                    covariance[i][j] /= denom;
                }
                covariance[i][i] += 1e-6; // Regularization for stability
            }
            return covariance;
        }
        
        // Uses Gauss-Jordan elimination to compute the inverse of a matrix
        private static double[][] invertMatrix(double[][] matrix) {
            int n = matrix.length;
            double[][] augmented = new double[n][2 * n];
            
            // Build augmented matrix [A | I]
            for (int i = 0; i < n; i++) {
                System.arraycopy(matrix[i], 0, augmented[i], 0, n);
                augmented[i][i + n] = 1.0;
            }
            
            // Apply row operations to transform [A | I] into [I | A^-1]
            for (int col = 0; col < n; col++) {
                // Find pivot
                int pivot = col;
                double max = Math.abs(augmented[pivot][col]);
                for (int row = col + 1; row < n; row++) {
                    double value = Math.abs(augmented[row][col]);
                    if (value > max) {
                         max = value; pivot = row; 
                    }
                }
                
                if (Math.abs(augmented[pivot][col]) < 1e-9) {
                    return null; // Matrix is singular
                }
                
                // Swap rows
                if (pivot != col) { 
                    double[] tmp = augmented[pivot]; augmented[pivot] = augmented[col]; augmented[col] = tmp; 
                }
                
                // Normalize pivot row
                double pivotVal = augmented[col][col];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[col][j] /= pivotVal;
                }
                
                // Eliminate other entries in the column
                for (int row = 0; row < n; row++) {
                    if (row == col) {
                        continue;
                    }
                    double factor = augmented[row][col];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[row][j] -= factor * augmented[col][j];
                    }
                }
            }
            
            // Extract the inverse matrix A^-1 from the right side
            double[][] inverse = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(augmented[i], n, inverse[i], 0, n);
            }
            return inverse;
        }
        
        // Calculates the Mahalanobis Distance for a given difference vector and inverse covariance
        private static double computeMahalanobisDistance(double[] diff, double[][] inverseCovariance) {
            // intermediate = Covariance^-1 * diff
            double[] intermediate = new double[diff.length];
            for (int i = 0; i < diff.length; i++) {
                double sum = 0;
                for (int j = 0; j < diff.length; j++) {
                    sum += inverseCovariance[i][j] * diff[j];
                }
                intermediate[i] = sum;
            }
            
            // distance = diff^T * intermediate
            double distance = 0;
            for (int i = 0; i < diff.length; i++) {
                distance += diff[i] * intermediate[i];
            }
            
            return Math.sqrt(Math.max(distance, 0));
        }
    }

    private static class AllAtOnce implements Algorithm {

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            Algorithm[] algorithms = {
                EUCLIDEAN_DISTANCE,
                MULTI_LAYER_PERCEPTRON,
                DISTANCE_FROM_CENTROID,
                new SupportVectorMachine(SupportVectorMachine.FeatureMode.ALL),
                new SupportVectorMachine(SupportVectorMachine.FeatureMode.CENTROID_ONLY),
                new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_CENTROID),
                new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_KMEANS),
                new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_GA),
                K_NEAREST_NEIGHBOUR,
                MAHALANOBIS_DISTANCE
            };

            // Votes for each digit class 0..9
            int[] votes = new int[10];

            for (Algorithm algorithm : algorithms) {
                try {
                    Object result = algorithm.predict(sample, trainingSet);

                    // SupportVectorMachine returns int[] (OneVsRest and OneVsOne). Count each prediction as a vote.
                    if (result instanceof int[]) {
                        int[] preds = (int[]) result;
                        for (int p : preds) {
                            if (p >= 0 && p < votes.length) votes[p]++;
                        }
                    }
                    // Other algorithms return a single Integer
                    else if (result instanceof Integer) {
                        int p = (Integer) result;
                        if (p >= 0 && p < votes.length) votes[p]++;
                    } else {
                        System.err.println("Unexpected result type from algorithm: " + algorithm.getClass().getSimpleName());
                    }
                } catch (Exception e) {
                    // Log the failure but continue with other algorithms
                    System.err.println("Error in algorithm: " + algorithm.getClass().getSimpleName());
                    e.printStackTrace();
                }
            }

            // Determine the majority-vote class
            int bestClass = 0;
            int maxVotes = -1;
            for (int i = 0; i < votes.length; i++) {
                if (votes[i] > maxVotes) {
                    maxVotes = votes[i]; bestClass = i; 
                }
            }
            return Integer.valueOf(bestClass);
        }
    }

    // ------------------------------------------------------------------------ 
    // --- EVALUATION FUNCTION ---
    // ------------------------------------------------------------------------

    // function to evaluate success rate of inputed algorithm
    private static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        
        // Start the animation in a separate thread
        Thread animationThread = new Thread(() -> showLoadingAnimation()); 
        animationThread.start();
        long startTime = System.nanoTime();

        // Ensure SVM is trained before evaluation begins
        if (algorithm instanceof SupportVectorMachine svm) {
            if (!svm.isTrained()) {
                 // Trigger training via a dummy prediction call on a training sample
                 svm.predict(dataSetA.get(0), dataSetA); 
            }
        }

        if (algorithm instanceof MultiLayerPerceptron) {
            System.out.println("\n--- " + label + " parameters used in calculation ---");
            System.out.println(((MultiLayerPerceptron) algorithm).getParameters());
        }

        int correctMatches = 0; // For single-prediction algorithms
        int correctOneVsRest = 0; // For SVM's One-vs-Rest result
        int correctOneVsOne = 0; // For SVM's One-vs-One result
        boolean isSplitResult = false; // Flag to check if we received SVM's split result

        // Iterate through all samples in the test set
        for (List<Integer> sample : dataSetB) {
            int actualDigit = sample.get(BITMAP_SIZE); // The correct answer
            Object result = algorithm.predict(sample, dataSetA);

            if (result instanceof int[]) {
                // Handle split result from SVM: [OneVsRest_Prediction, OneVsOne_Prediction]
                isSplitResult = true;
                int[] preds = (int[]) result;
                if (preds[0] == actualDigit) {
                    correctOneVsRest++;
                }
                if (preds[1] == actualDigit) {
                    correctOneVsOne++;
                }
            } else if (result instanceof Integer) {
                // Handle single Integer prediction from other algorithms
                if ((Integer) result == actualDigit) {
                    correctMatches++;
                }
            }
        }

        // Stop the animation
        animationThread.interrupt();
        long endTime = System.nanoTime();
        double duration = (endTime - startTime) / 1_000_000_000.0;
        int total = dataSetB.size();

        System.out.println("\n");
        System.out.println("\n--- " + label + " Success Rate ---");
        if (isSplitResult) {
            // Print results for SVM variants
            System.out.printf("   One-vs-Rest Correct: %d / %d%n", correctOneVsRest, total);
            System.out.printf("   One-vs-Rest Success Rate: %.5f%%%n", (correctOneVsRest/(double)total)*100);
            System.out.printf("   One-vs-One Correct: %d / %d%n", correctOneVsOne, total);
            System.out.printf("   One-vs-One Success Rate: %.5f%%%n", (correctOneVsOne/(double)total)*100);
        } else {
            // Print results for single-prediction algorithms
            System.out.printf("   Correct Matches: %d / %d%n", correctMatches, total);
            System.out.printf("   Success Rate: %.5f%%%n", (correctMatches/(double)total)*100);
        }
        System.out.println("   Evaluation Time: " + duration + " seconds");
        System.out.println("\n");
    }

    private static void showLoadingAnimation() {
        try {
            String[] frames = {".  ", ".. ", "..."};
            while (true) {
                for (String frame : frames) { System.out.print("\revaluating" + frame); Thread.sleep(500); }
                System.out.print("\r             "); Thread.sleep(500); // Clear the line temporarily
            }
        } catch (InterruptedException e) { 
            // When interrupted, animation thread exits gracefully
            Thread.currentThread().interrupt(); 
            System.out.print("\r"); // Clear the final animation frame
        }
    }

    // ------------------------------------------------------------------------
    // --- CSV READING AND PRINTING FUNCTIONS ---
    // ------------------------------------------------------------------------

    // Function to read the dataset from a CSV file
    private static List<List<Integer>> readCsvFile(String dataSetFilePath) {
        List<List<Integer>> dataSet = new ArrayList<>();
        System.out.println("Reading CSV: " + dataSetFilePath);
        try (Scanner scanner = new Scanner(new java.io.File(dataSetFilePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] values = line.split(",");

                if (values.length != ENTIRE_BITMAP_SIZE) {
                    continue; // Skip malformed rows
                }

                List<Integer> currentRow = new ArrayList<>();
                boolean error = false;

                for (int i = 0; i < ENTIRE_BITMAP_SIZE; i++) {
                    try { 
                        currentRow.add(Integer.parseInt(values[i].trim())); 
                    }
                    catch (NumberFormatException e) {
                        error = true; 
                        break; 
                    }
                }
                if (!error) { 
                    dataSet.add(currentRow); 
                }
            }
            return dataSet;
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            return null;
        }
    }

    private static void printDataSet(List<List<Integer>> dataSet) {
        System.out.println("--- Entire Dataset ---");
        for (int i = 0; i < dataSet.size(); i++) printRow(i, dataSet.get(i));
    }
    private static void printLimitedDataSet(List<List<Integer>> dataSet) {
        System.out.println("--- First " + BITMAPS_TO_DISPLAY + " Samples ---");
        for (int i = 0; i < Math.min(BITMAPS_TO_DISPLAY, dataSet.size()); i++) printRow(i, dataSet.get(i));
    }
    
    // Prints a single row's pixel values and digit label
    private static void printRow(int i, List<Integer> row) {
        // The last element is the digit label
        int digitLabel = row.get(BITMAP_SIZE); 
        
        System.out.print("Sample " + (i + 1) + " (Digit: " + digitLabel + "): [");
        // Print the 64 pixel values
        for (int j = 0; j < BITMAP_SIZE; j++) {
            System.out.print(row.get(j));
            if (j < BITMAP_SIZE - 1) System.out.print(", ");
        }
        System.out.println("]");
    }

    private static void PrintDataUserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB, Scanner scanner) {
        boolean running = true;
        while (running) {
            System.out.println("\n=== Print Actions: ===");
            System.out.println("1 -> Print entire A");
            System.out.println("2 -> Print entire B");
            System.out.println("3 -> Print subset A (First " + BITMAPS_TO_DISPLAY + ")");
            System.out.println("4 -> Print subset B (First " + BITMAPS_TO_DISPLAY + ")");
            System.out.println("0 -> Exit");
            System.out.print("Choose bettween 0-4: ");
            try {
                int choice = scanner.nextInt();
                switch (choice) {
                    case 1: if(dataSetA!=null) printDataSet(dataSetA); break;
                    case 2: if(dataSetB!=null) printDataSet(dataSetB); break;
                    case 3: if(dataSetA!=null) printLimitedDataSet(dataSetA); break;
                    case 4: if(dataSetB!=null) printLimitedDataSet(dataSetB); break;
                    case 0: running = false; break;
                    default: System.out.println("\nInvalid choice. Please enter a number corresponding to available actions.");
                }
            } catch (Exception e) { 
                System.out.println("\nInvalid input. Please enter a number.");
                scanner.nextLine(); // Consume the invalid input
            }
        }
    }

    private static void UserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB) {
        Scanner scanner = new Scanner(System.in); 
        boolean running = true;
        
        if (dataSetA == null || dataSetB == null) {
            System.err.println("Cannot run program: Dataset loading failed.");
            return;
        }
        
        while (running) {
            System.out.println("\n=== Actions: ===");
            System.out.println("1 -> Print Options");
            System.out.println("2 -> Euclidean Distance");
            System.out.println("3 -> Multi Layer Perceptron");
            System.out.println("4 -> Distance From Centroid");
            System.out.println("5 -> Support Vector Machine");
            System.out.println("6 -> K Nearest Neighbour");
            System.out.println("7 -> Mahalanobis Distance");
            System.out.println("8 -> All at Once");
            System.out.println("0 -> Exit");
            System.out.print("Choose bettween 0-8: ");
            
            try {
                int choice = scanner.nextInt();
                switch (choice) {
                    case 1: 
                        PrintDataUserInterface(dataSetA, dataSetB, scanner); 
                        break;
                        
                    case 2: 
                        evaluateAlgorithm(dataSetA, dataSetB, EUCLIDEAN_DISTANCE, "Euclidean Distance"); // train on A, test on B
                        break;
                        
                    case 3: 
                        evaluateAlgorithm(dataSetA, dataSetB, MULTI_LAYER_PERCEPTRON, "Multi Layer Perceptron"); // train on A, test on B
                        break;
                        
                    case 4: 
                        evaluateAlgorithm(dataSetA, dataSetB, DISTANCE_FROM_CENTROID, "Distance From Centroid"); // train on A, test on B
                        break;
                        
                    case 5:
                        // Baseline: Centroid Distances only
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.CENTROID_ONLY), "SVM [Centroid Only]"); // train on A, test on B
                        
                        // Full combined feature set (Raw + Centroid + KMeans + GA)
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.ALL), "SVM [Combined Features]"); // train on A, test on B
                        
                        // Simple Raw + Centroid mix
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_CENTROID), "SVM [Raw + Centroid]"); // train on A, test on B
                        
                        // Raw + K-Means Distances
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_KMEANS), "SVM [Raw + KMeans]"); // train on A, test on B
                        
                        // Raw + GA Weighted Pixels
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_GA), "SVM [Raw + GA]"); // train on A, test on B
                        break;
                        
                    case 6: 
                        evaluateAlgorithm(dataSetA, dataSetB, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour"); // train on A, test on B
                        break;
                        
                    case 7:
                        evaluateAlgorithm(dataSetA, dataSetB, MAHALANOBIS_DISTANCE, "Mahalanobis Distance"); // train on A, test on B
                        break;

                    case 8:
                        evaluateAlgorithm(dataSetA, dataSetB, ALL_AT_ONCE, "All at Once (Ensemble)"); // train on A, test on B
                        break;

                    case 0:
                        System.out.println("\nExiting");
                        running = false;
                        break;
                        
                    default:
                        System.out.println("\nInvalid choice. Please enter a number corresponding to available actions.");
                }
                
            } catch (Exception error) {
                System.out.println("\nInvalid input. Please enter a number corresponding to available actions.");
                scanner.nextLine(); // Clear the invalid input
            }
        }
        
        scanner.close();
    }
    
	public static void main(String[] args) {
        // read datasets
	    List<List<Integer>> dataSetA = readCsvFile(DATASET_A_FILE_PATH);
	    List<List<Integer>> dataSetB = readCsvFile(DATASET_B_FILE_PATH);
	    
        // start user interface
	    UserInterface(dataSetA, dataSetB);
	}
}