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
    // The MLP is instantiated dynamically in the UI to test different kernel functions
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
            for (int iteration = 0; iteration < BITMAP_SIZE; iteration++) {
                sumPerClass[instance][iteration] += row.get(iteration);
            }
        }

        // Calculate the average (centroid)
        for (int digit = 0; digit < 10; digit++) {
            if (countPerClass[digit] > 0) {
                for (int iteration = 0; iteration < BITMAP_SIZE; iteration++) {
                    centroids[digit][iteration] = sumPerClass[digit][iteration] / countPerClass[digit];
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
        for (int iteration = 0; iteration < BITMAP_SIZE; iteration++) {
            centroids[0][iteration] = first.get(iteration);
        }

        for (int cluster = 1; cluster < clusters; cluster++) {
            double[] dist = new double[trainingSet.size()];
            double total = 0;
            // Calculate distance to the nearest existing centroid for all samples
            for (int i = 0; i < trainingSet.size(); i++) {
                List<Integer> sample = trainingSet.get(i);
                double minDistance = Double.MAX_VALUE;
                for (int c = 0; c < cluster; c++) {
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
                        centroids[cluster][j] = picked.get(j);
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
                double bestDouble = Double.MAX_VALUE;
                for (int c = 0; c < clusters; c++) {
                    double d = 0;
                    for (int f = 0; f < BITMAP_SIZE; f++) {
                        double diff = s.get(f) - centroids[c][f];
                        d += diff * diff;
                    }
                    if (d < bestDouble) {
                        bestDouble = d;
                        best = c;
                    }
                }
                counts[best]++;
                for (int f = 0; f < BITMAP_SIZE; f++)
                    sums[best][f] += s.get(f);
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

    // function to compute Euclidean distances from a sample to the K-Means centroiEUCLIDEAN_DISTANCEds
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
            for (int f = 0; f < BITMAP_SIZE; f++)
                individual[f] = randomGenerator.nextDouble();
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
            if (bestScore >= fitnessThreshold) {
                break;
            }

            // keep the best, replace others with children
            List<double[]> nextGeneration = new ArrayList<>();
            nextGeneration.add(bestIndividual.clone()); // Keep best
            while (nextGeneration.size() < population.size()) {
                // Randomly select two parents (excluding best)
                int parent1Idx = randomGenerator.nextInt(population.size());
                int parent2Idx = randomGenerator.nextInt(population.size());
                if (parent1Idx == bestIndex) {
                    parent1Idx = (parent1Idx + 1) % population.size();
                }
                if (parent2Idx == bestIndex) {
                    parent2Idx = (parent2Idx + 1) % population.size();
                }
                double[] parent1 = population.get(parent1Idx);
                double[] parent2 = population.get(parent2Idx);
                // Crossover
                double[] childIndividual = new double[BITMAP_SIZE];
                for (int f = 0; f < BITMAP_SIZE; f++) {
                    childIndividual[f] = randomGenerator.nextDouble() < 0.5 ? parent1[f] : parent2[f];
                }
                    
                // Mutation
                for (int f = 0; f < BITMAP_SIZE; f++) {
                    if (randomGenerator.nextDouble() < GA_MUTATION) {
                         childIndividual[f] += randomGenerator.nextGaussian() * 0.1;
                    }
                }
                   
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
            int bestClass = 0;
            double bestDistance = Double.MAX_VALUE;
            // Classify sample using weighted distance to centroids
            for (int c = 0; c < centroids.length; c++) {
                double distance = 0;
                for (int f = 0; f < BITMAP_SIZE; f++) {
                    // Apply weight before calculating difference
                    double diff = (sample.get(f) * weights[f]) - centroids[c][f];
                    distance += diff * diff;
                }
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClass = c;
                }
            }
            if (bestClass == sample.get(BITMAP_SIZE)) {
                correctMatches++;
            }
        }
        return correctMatches / (double) Math.max(1, trainingSet.size());
    }

    // Convert raw pixel values to a double vector
    public static double[] buildRawPixelsVector(List<Integer> sample) {
        double[] vector = new double[BITMAP_SIZE];
        for (int i = 0; i < BITMAP_SIZE; i++) {
             vector[i] = sample.get(i);
        }
        return vector;
    }

    // Create a feature vector of distances to the 10 class centroids
    public static double[] buildCentroidDistanceVector(List<Integer> sample, double[][] centroids) {
        int n = (centroids != null) ? centroids.length : 0;
        double[] vector = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int f = 0; f < BITMAP_SIZE; f++) {
                double diff = sample.get(f) - centroids[i][f];
                sum += diff * diff;
            }
            vector[i] = Math.sqrt(sum);
        }
        return vector;
    }

    // Create a feature vector of distances to the K-Means cluster centers
    public static double[] buildKMeansDistanceVector(List<Integer> sample, double[][] kmeansCentroids) {
        if (kmeansCentroids == null) {
            return new double[0];
        }
        int clusters = kmeansCentroids.length;
        double[] vector = new double[clusters];
        for (int i = 0; i < clusters; i++) {
            double sum = 0;
            for (int j = 0; j < BITMAP_SIZE; j++) {
                double diff = sample.get(j) - kmeansCentroids[i][j];
                sum += diff * diff;
            }
            vector[i] = Math.sqrt(sum);
        }
        return vector;
    }

    // Create a feature vector of raw pixels multiplied by the Genetic Algorithm weights
    public static double[] buildGAWeightedVector(List<Integer> sample, double[] gaWeights) {
        double[] vector = new double[BITMAP_SIZE];
        for (int i = 0; i < BITMAP_SIZE; i++) {
            vector[i] = sample.get(i) * (gaWeights != null ? gaWeights[i] : 1.0);
        }
        return vector;
    }

    // Utility function to combine multiple feature vectors into one long vector
    public static double[] concatVectors(double[]... parts) {
        int total = 0;
        for (double[] p : parts) {
            if (p != null) {
                total += p.length;
            }
        }
        double[] out = new double[total];
        int pos = 0;
        for (double[] p : parts) {
            if (p == null) {
                continue;
            }
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
            if (trainingSet == null || trainingSet.isEmpty()) {
                throw new IllegalArgumentException("Dataset empty");
            }
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

   // Multi Layer Perceptron Algorithm
    private static class MultiLayerPerceptron implements Algorithm {
        // Enum to control which features to use
        public enum FeatureMode {
            RAW_ONLY,           // Just the 64 raw pixels
            CENTROID_ONLY,      // Only distances to class centroids
            RAW_CENTROID,       // Raw pixels + centroid distances
            RAW_KMEANS,         // Raw pixels + K-Means distances
            RAW_GA,             // Raw pixels + GA weighted pixels
            ALL                 // Raw + Centroid + KMeans + GA
        }

        // Hyperparameters
        private static final int PERCEPTRONS = 500;
        private static final int EPOCHS = 100;
        // 500 perceptrons and 500 epochs have best results when training on A and testing on B
        // 1000 perceptrons and 50 epochs have best results when training on B and testing on A
        private static final double LEARNING_RATE = 0.1;
        private static final long RANDOM_SEED = 42; // fixed seed for reproducibility
        private static final int CLASSES = 10;

        private final FeatureMode mode;
        
        // Network weights
        private double[][] weightsInputHidden; // weights from input to hidden layer [PERCEPTRONS][inputSize]
        private double[] biasHidden;
        private double[][] weightsHiddenOutput; // weights from hidden to output layer [CLASSES][PERCEPTRONS]
        private double[] biasOutput;
        private boolean trained = false;
        
        // Feature caches (computed once during training)
        private double[][] centroidCache;
        private double[][] kmeansCentroidCache;
        private double[] gaWeightsCache;
        private int inputSize; // Actual input size

        private double[] featureMean;
        private double[] featureStd;

        // Default constructor uses raw pixels only (backward compatible)
        public MultiLayerPerceptron() {
            this(FeatureMode.RAW_ONLY);
        }
        
        // Constructor with feature mode
        public MultiLayerPerceptron(FeatureMode mode) {
            this.mode = mode;
        }

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (!trained) {
                train(trainingSet);
            }
            
            double[] inputs = buildFeatureVector(sample); // create all beatmaps with selected features

            // normalize bitmaps with exclusion oforiginal raw bitmap
            if (mode != FeatureMode.RAW_ONLY){
                inputs = normalizeFeatures(inputs);
            }
            
            double[] outputs = forward(inputs); // forward pass
            
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

        private void train(List<List<Integer>> trainingSet) {
            // Compute feature caches based on mode
            computeFeatureCaches(trainingSet);
            
            // Determine input size from feature vector
            inputSize = buildFeatureVector(trainingSet.get(0)).length;
            
            // compute normalization of bitmaps with exclusion oforiginal raw bitmap
            if (mode != FeatureMode.RAW_ONLY){
                computeNormalizationStats(trainingSet);
            }
            

            initializeWeights();

            // Training loop
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                for (List<Integer> row : trainingSet) {
                    double[] inputs = buildFeatureVector(row);
                    // normalize bitmaps with exclusion of original raw bitmap
                    if (mode != FeatureMode.RAW_ONLY) {
                        inputs = normalizeFeatures(inputs);
                    }
                    int targetClass = row.get(BITMAP_SIZE);
                    double[] hidden = new double[PERCEPTRONS];
                    double[] outputs = new double[CLASSES];

                    // Forward pass - hidden layer
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        double sum = biasHidden[h];
                        for (int i = 0; i < inputSize; i++) {
                            sum += weightsInputHidden[h][i] * inputs[i];
                        }
                        hidden[h] = sigmoid(sum);
                    }

                    // Forward pass - output layer
                    for (int o = 0; o < CLASSES; o++) {
                        double sum = biasOutput[o];
                        for (int h = 0; h < PERCEPTRONS; h++) {
                            sum += weightsHiddenOutput[o][h] * hidden[h];
                        }
                        outputs[o] = sigmoid(sum);
                    }

                    // Prepare target vector
                    double[] target = new double[CLASSES];
                    target[targetClass] = 1.0;

                    // Backpropagation - output layer
                    double[] outputDeltas = new double[CLASSES];
                    for (int o = 0; o < CLASSES; o++) {
                        double error = target[o] - outputs[o];
                        outputDeltas[o] = error * sigmoidDerivative(outputs[o]);
                    }

                    // Backpropagation - hidden layer
                    double[] hiddenDeltas = new double[PERCEPTRONS];
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        double error = 0;
                        for (int o = 0; o < CLASSES; o++) {
                            error += outputDeltas[o] * weightsHiddenOutput[o][h];
                        }
                        hiddenDeltas[h] = error * sigmoidDerivative(hidden[h]);
                    }

                    // Update weights - input to hidden
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        for (int i = 0; i < inputSize; i++) {
                            weightsInputHidden[h][i] += LEARNING_RATE * hiddenDeltas[h] * inputs[i];
                        }
                        biasHidden[h] += LEARNING_RATE * hiddenDeltas[h];
                    }

                    // Update weights - hidden to output
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

        // Compute necessary feature caches based on the selected mode
        private void computeFeatureCaches(List<List<Integer>> trainingSet) {
            boolean needCentroids = (mode == FeatureMode.ALL || mode == FeatureMode.CENTROID_ONLY || 
                                    mode == FeatureMode.RAW_CENTROID);
            boolean needKMeans = (mode == FeatureMode.ALL || mode == FeatureMode.RAW_KMEANS);
            boolean needGA = (mode == FeatureMode.ALL || mode == FeatureMode.RAW_GA);

            if (needCentroids) {
                centroidCache = calculateCentroids(trainingSet);
            }
            if (needKMeans) {
                kmeansCentroidCache = computeKMeansCentroids(trainingSet, KMEANS_CLUSTERS);
            }
            if (needGA) {
                gaWeightsCache = evolveGeneticWeights(trainingSet);
            }
        }

        // Build feature vector based on selected mode
        private double[] buildFeatureVector(List<Integer> sample) {
            double[] raw = buildRawPixelsVector(sample);

            switch (mode) {
                case RAW_ONLY:
                    return raw;
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

        // Compute mean and standard deviation for feature normalization
        private void computeNormalizationStats(List<List<Integer>> trainingSet) {
            featureMean = new double[inputSize];
            featureStd = new double[inputSize];
            
            // Compute mean
            for (List<Integer> row : trainingSet) {
                double[] features = buildFeatureVector(row);
                for (int i = 0; i < inputSize; i++) {
                    featureMean[i] += features[i];
                }
            }
            for (int i = 0; i < inputSize; i++) {
                featureMean[i] /= trainingSet.size();
            }
            
            // Compute standard deviation
            for (List<Integer> row : trainingSet) {
                double[] features = buildFeatureVector(row);
                for (int i = 0; i < inputSize; i++) {
                    double diff = features[i] - featureMean[i];
                    featureStd[i] += diff * diff;
                }
            }
            for (int i = 0; i < inputSize; i++) {
                featureStd[i] = Math.sqrt(featureStd[i] / trainingSet.size());
                // Prevent division by zero
                if (featureStd[i] < 1e-9) {
                    featureStd[i] = 1.0;
                }
            }
        }

        // Normalize features using computed statistics
        private double[] normalizeFeatures(double[] features) {
            double[] normalized = new double[features.length];
            for (int i = 0; i < features.length; i++) {
                normalized[i] = (features[i] - featureMean[i]) / featureStd[i];
            }
            return normalized;
        }

        private void initializeWeights() {
            weightsInputHidden = new double[PERCEPTRONS][inputSize];
            biasHidden = new double[PERCEPTRONS];
            weightsHiddenOutput = new double[CLASSES][PERCEPTRONS];
            biasOutput = new double[CLASSES];
            Random random = new Random(RANDOM_SEED);

            double range = 0.1;
            for (int h = 0; h < PERCEPTRONS; h++) {
                biasHidden[h] = (random.nextDouble() * 2 - 1) * range;
                for (int i = 0; i < inputSize; i++) {
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

        private double[] forward(double[] inputs) {
            double[] hidden = new double[PERCEPTRONS];
            for (int h = 0; h < PERCEPTRONS; h++) {
                double sum = biasHidden[h];
                for (int i = 0; i < inputSize; i++) {
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

        private double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        private double sigmoidDerivative(double activatedValue) {
            return activatedValue * (1.0 - activatedValue);
        }

        public String getParameters() {
            return "Mode: " + mode + ", Epochs: " + EPOCHS + ", Learning rate: " + LEARNING_RATE + ", Perceptrons: " + PERCEPTRONS + ", Input size: " + inputSize + ", Random seed: " + RANDOM_SEED;
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
            ALL,                // Raw + Centroid + KMeans + GA (most complex kernel)
            CENTROID_ONLY,      // Only Distances to Class Centroids
            RAW_CENTROID,       // Raw Pixels + Centroid Distances
            RAW_KMEANS,         // Raw Pixels + K-Means Distances
            RAW_GA              // Raw Pixels + GA Weighted Pixels
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
            return new int[] { oneVsRestPrediction, oneVsOnePrediction };
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
            if (needCentroids) {
                centroidCache = calculateCentroids(trainingSet);
            }
            if (needKMeans) {
                kmeansCentroidCache = computeKMeansCentroids(trainingSet, KMEANS_CLUSTERS);
            }
            if (needGA) {
                gaWeightsCache = evolveGeneticWeights(trainingSet);
            }

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
                    for (int k = 0; k < n; k++) {
                        if (labels[k] == i || labels[k] == j) {
                            pairIndices.add(k);
                        }
                    }

                    if (!pairIndices.isEmpty()) {
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
            for (int i = 0; i < n; i++) {
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
                            bestRivalScore = res.scores[c];
                            bestRival = c;
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
                        for (int feature = 0; feature < featureSize; feature++) {
                            model.weights[0][feature] += LEARNING_RATE * y * features[idx][feature];
                        }
                        model.bias[0] += LEARNING_RATE * y;
                        updated = true;
                    }
                    model.accumulateAverages();
                }
                if (!updated) {
                    break;
                }
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
                for (int class = 0; class < weights.length; class++) {
                    scores[class] = pointInFeatureSpace(class, features) + bias[class];
                    if (scores[class] > max) {
                        max = scores[class];
                        predicted = class;
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

                for (int point = 0; point < features.length; point++) {
                    sum += weights[classIdx][point] * features[point];
                }

                return sum;
            }

            // Applies the learning update to the weights and bias for a given class
            void update(double[] features, int classIdx, double direction) {
                for (int i = 0; i < features.length; i++) {
                    weights[classIdx][i] += LEARNING_RATE * direction * features[i];
                }
                bias[classIdx] += LEARNING_RATE * direction;
            }

            // Adds current weights/biases to the running sums for averaging
            void accumulateAverages() {
                steps++;
                for (int classIdx = 0; classIdx < weights.length; classIdx++) {
                    for (int f = 0; f < weights[0].length; f++) {
                        weightSums[classIdx][f] += weights[classIdx][f];
                    }
                    biasSums[classIdx] += bias[classIdx];
                }
            }

            // Calculates the final averaged weights
            void finalizeWeights() {
                if (steps == 0) {
                    return;
                }
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < weights[0].length; f++) {
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
                if (matrix.length == 0) {
                    return;
                }

                // Calculates Mean and Standard Deviation (Fit)
                for (int feature = 0; feature < featureSize; feature++) {
                    double sum = 0, sumSq = 0;
                    for (double[] row : matrix) {
                        sum += row[feature];
                        sumSq += row[feature] * row[feature];
                    }
                    mean[feature] = sum / matrix.length;
                    double var = (sumSq / matrix.length) - (mean[feature] * mean[feature]);
                    // Use a small epsilon to prevent division by zero for constant features
                    std[feature] = Math.max(Math.sqrt(Math.max(var, 0)), 1e-9);
                }

                // Apply Normalization (Transform)
                for (double[] row : matrix) {
                    for (int feature = 0; feature < featureSize; feature++) {
                        row[feature] = (row[feature] - mean[feature]) / std[feature];
                    }
                }
            }

            // Applies normalization to a single input vector using pre-calculated stats
            double[] normalize(double[] vector) {
                double[] out = new double[featureSize];
                for (int feature = 0; feature < featureSize; feature++)
                    out[feature] = (vector[feature] - mean[feature]) / std[feature];
                return out;
            }
        }

        // Inner class: Simple data structure to hold results from computeScores
        private static class PredictionResult {
            int predictedClass;
            double[] scores;

            PredictionResult(int predicted, double[] score) {
                this.predictedClass = predicted;
                this.scores = score;
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
                for (int row = 0; row < BITMAP_SIZE; row++) {
                    double diff = sample.get(row) - candidate.get(row);
                    distance += diff * diff; // Squared Euclidean Distance gives better results than square root
                }

                // Keep only the K smallest distances
                if (heap.size() < K) {
                    heap.offer(new double[] { distance, candidate.get(BITMAP_SIZE) });
                } else if (distance < heap.peek()[0]) {
                    heap.poll();
                    heap.offer(new double[] { distance, candidate.get(BITMAP_SIZE) });
                }
            }

            // count votes for each class among the K nearest neighbors
            int[] votes = new int[10];
            while (!heap.isEmpty()) {
                int digit = (int) heap.poll()[1];
                votes[digit]++;
            }

            // Return the majority class
            int bestDigit = 0;
            int bestVotes = 0;
            for (int digit = 0; digit < votes.length; digit++) {
                if (votes[digit] > bestVotes) {
                    bestVotes = votes[digit];
                    bestDigit = digit;
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
                    bestDistance = distance;
                    bestClass = digit;
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
                        max = value;
                        pivot = row;
                    }
                }

                if (Math.abs(augmented[pivot][col]) < 1e-9) {
                    return null; // Matrix is singular
                }

                // Swap rows
                if (pivot != col) {
                    double[] tmp = augmented[pivot];
                    augmented[pivot] = augmented[col];
                    augmented[col] = tmp;
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
            return Math.sqrt(distance);
        }
    }

    // All at Once Algorithm (Ensemble of all algorithms with majority voting)
    private static class AllAtOnce implements Algorithm {
        
        // Pre-trained SVM instances (trained once, used many times)
        private SupportVectorMachine svmCentroidOnly;
        private SupportVectorMachine svmAll;
        private SupportVectorMachine svmRawCentroid;
        private SupportVectorMachine svmRawKMeans;
        private SupportVectorMachine svmRawGA;
        
        // Pre-trained MLP instances (trained once, used many times)
        private MultiLayerPerceptron mlpRawOnly;
        private MultiLayerPerceptron mlpCentroidOnly;
        private MultiLayerPerceptron mlpAll;
        private MultiLayerPerceptron mlpRawCentroid;
        private MultiLayerPerceptron mlpRawKMeans;
        private MultiLayerPerceptron mlpRawGA;
        
        private boolean trained = false;
        
        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            // Train all models once on first call
            if (!trained) {
                System.out.println("Training SVM and MLP variants (this will take a moment)...");
                trainAllSVMs(trainingSet);
                trainAllMLPs(trainingSet);
                trained = true;
                System.out.println("variants training complete!");
            }
            
            Algorithm[] algorithms = {
                EUCLIDEAN_DISTANCE,
                DISTANCE_FROM_CENTROID,
                K_NEAREST_NEIGHBOUR,
                MAHALANOBIS_DISTANCE,
                // SVM variants
                svmCentroidOnly,
                svmAll,
                svmRawCentroid,
                svmRawKMeans,
                svmRawGA,
                // MLP variants
                mlpRawOnly,
                mlpCentroidOnly,
                mlpAll,
                mlpRawCentroid,
                mlpRawKMeans,
                mlpRawGA
            };

            // Array to store votes for each digit class (0-9)
            int[] votes = new int[10];

            // Run each algorithm and collect predictions
            for (Algorithm algorithm : algorithms) {
                Object result = algorithm.predict(sample, trainingSet);
                
                // Handle algorithms that return int[] (e.g., SupportVectorMachine)
                if (result instanceof int[]) {
                    int[] intResult = (int[]) result;
                    // SVM returns [OneVsRest, OneVsOne], count both predictions
                    for (int prediction : intResult) {
                        if (prediction >= 0 && prediction < 10) {
                            votes[prediction]++;
                        }
                    }
                } 
                // Handle algorithms that return Integer
                else if (result instanceof Integer) {
                    int prediction = (Integer) result;
                    if (prediction >= 0 && prediction < 10) {
                        votes[prediction]++;
                    }
                }
            }

            int maxVotes = 0;
            int bestDigit = 0;
            
            for (int digit = 0; digit < votes.length; digit++) {
                if (votes[digit] > maxVotes) {
                    maxVotes = votes[digit];
                    bestDigit = digit;
                }
            }

            // Return the digit with the most votes
            return bestDigit;
        }
        
        // Train all SVM variants once
        private void trainAllSVMs(List<List<Integer>> trainingSet) {
            System.out.println("  Training SVM variants...");
            
            svmCentroidOnly = new SupportVectorMachine(SupportVectorMachine.FeatureMode.CENTROID_ONLY);
            svmCentroidOnly.predict(trainingSet.get(0), trainingSet); // Trigger training
            
            svmAll = new SupportVectorMachine(SupportVectorMachine.FeatureMode.ALL);
            svmAll.predict(trainingSet.get(0), trainingSet);
            
            svmRawCentroid = new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_CENTROID);
            svmRawCentroid.predict(trainingSet.get(0), trainingSet);
            
            svmRawKMeans = new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_KMEANS);
            svmRawKMeans.predict(trainingSet.get(0), trainingSet);
            
            svmRawGA = new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_GA);
            svmRawGA.predict(trainingSet.get(0), trainingSet);
            
            System.out.println("  SVM training complete!");
        }

        // Train all MLP variants once
        private void trainAllMLPs(List<List<Integer>> trainingSet) {
            System.out.println("  Training MLP variants...");
            
            mlpRawOnly = new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_ONLY);
            mlpRawOnly.predict(trainingSet.get(0), trainingSet); // Trigger training
            
            mlpCentroidOnly = new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.CENTROID_ONLY);
            mlpCentroidOnly.predict(trainingSet.get(0), trainingSet);
            
            mlpAll = new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.ALL);
            mlpAll.predict(trainingSet.get(0), trainingSet);
            
            mlpRawCentroid = new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_CENTROID);
            mlpRawCentroid.predict(trainingSet.get(0), trainingSet);
            
            mlpRawKMeans = new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_KMEANS);
            mlpRawKMeans.predict(trainingSet.get(0), trainingSet);
            
            mlpRawGA = new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_GA);
            mlpRawGA.predict(trainingSet.get(0), trainingSet);
            
            System.out.println("  MLP training complete!");
        }
        
        // Accessor methods for pretrained models (for use in runAllInOrder)
        public SupportVectorMachine getSvmCentroidOnly() { return svmCentroidOnly; }
        public SupportVectorMachine getSvmAll() { return svmAll; }
        public SupportVectorMachine getSvmRawCentroid() { return svmRawCentroid; }
        public SupportVectorMachine getSvmRawKMeans() { return svmRawKMeans; }
        public SupportVectorMachine getSvmRawGA() { return svmRawGA; }
        
        public MultiLayerPerceptron getMlpRawOnly() { return mlpRawOnly; }
        public MultiLayerPerceptron getMlpCentroidOnly() { return mlpCentroidOnly; }
        public MultiLayerPerceptron getMlpAll() { return mlpAll; }
        public MultiLayerPerceptron getMlpRawCentroid() { return mlpRawCentroid; }
        public MultiLayerPerceptron getMlpRawKMeans() { return mlpRawKMeans; }
        public MultiLayerPerceptron getMlpRawGA() { return mlpRawGA; }
    }

    // Data structure to hold evaluation results    
    private static class EvaluationResult {
        String algorithmName;
        int correctMatches;
        int total;
        double successRate;
        double evaluationTime;
        
        // For SVM results
        int correctOneVsRest;
        int correctOneVsOne;
        double successRateOneVsRest;
        double successRateOneVsOne;
        boolean isSplitResult;
        
        EvaluationResult(String name, int correct, int total, double time) {
            this.algorithmName = name;
            this.correctMatches = correct;
            this.total = total;
            this.successRate = (correct / (double) total) * 100;
            this.evaluationTime = time;
            this.isSplitResult = false;
        }
        
        EvaluationResult(String name, int correctOvR, int correctOvO, int total, double time) {
            this.algorithmName = name;
            this.correctOneVsRest = correctOvR;
            this.correctOneVsOne = correctOvO;
            this.total = total;
            this.successRateOneVsRest = (correctOvR / (double) total) * 100;
            this.successRateOneVsOne = (correctOvO / (double) total) * 100;
            this.evaluationTime = time;
            this.isSplitResult = true;
        }
    }

    // function to evaluate success rate of inputed algorithm
    private static EvaluationResult evaluateAlgorithmWithResults(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        long startTime = System.nanoTime();

        // Ensure SVM is trained before evaluation begins
        if (algorithm instanceof SupportVectorMachine svm) {
            if (!svm.isTrained()) {
                svm.predict(dataSetA.get(0), dataSetA);
            }
        }

        if (algorithm instanceof MultiLayerPerceptron) {
            System.out.println("\n--- " + label + " parameters used in calculation ---");
            System.out.println(((MultiLayerPerceptron) algorithm).getParameters());
        }

        int correctMatches = 0;
        int correctOneVsRest = 0;
        int correctOneVsOne = 0;
        boolean isSplitResult = false;

        for (List<Integer> sample : dataSetB) {
            int actualDigit = sample.get(BITMAP_SIZE);
            Object result = algorithm.predict(sample, dataSetA);

            if (result instanceof int[]) {
                isSplitResult = true;
                int[] preds = (int[]) result;
                if (preds[0] == actualDigit) {
                    correctOneVsRest++;
                }
                if (preds[1] == actualDigit) {
                    correctOneVsOne++;
                }
            } else if (result instanceof Integer) {
                if ((Integer) result == actualDigit) {
                    correctMatches++;
                }
            }
        }

        long endTime = System.nanoTime();
        double duration = (endTime - startTime) / 1_000_000_000.0;
        int total = dataSetB.size();

        System.out.println("\n--- " + label + " Success Rate ---");
        if (isSplitResult) {
            System.out.printf("   One-vs-Rest Correct: %d / %d%n", correctOneVsRest, total);
            System.out.printf("   One-vs-Rest Success Rate: %.5f%%%n", (correctOneVsRest / (double) total) * 100);
            System.out.printf("   One-vs-One Correct: %d / %d%n", correctOneVsOne, total);
            System.out.printf("   One-vs-One Success Rate: %.5f%%%n", (correctOneVsOne / (double) total) * 100);
            return new EvaluationResult(label, correctOneVsRest, correctOneVsOne, total, duration);
        } else {
            System.out.printf("   Correct Matches: %d / %d%n", correctMatches, total);
            System.out.printf("   Success Rate: %.5f%%%n", (correctMatches / (double) total) * 100);
            System.out.println("   Evaluation Time: " + duration + " seconds");
            System.out.println("\n");
            return new EvaluationResult(label, correctMatches, total, duration);
        }
    }

    // private static void showLoadingAnimation() {
    //     try {
    //         String[] frames = { ".  ", ".. ", "..." };
    //         String baseText = "evaluating";

    //         while (true) {
    //             for (String frame : frames) {
    //                 System.out.print("\r" + baseText + frame);
    //                 Thread.sleep(100000);
    //             }
    //         }
    //     } catch (InterruptedException e) {
    //         // When interrupted, animation thread exits gracefully.
    //         Thread.currentThread().interrupt();
            
    //         // to move the cursor to a fresh line for the main thread's output.
    //         System.out.print("\r" + "                              " + "\r");
    //     }
    // }

    // Wrapper function to call evaluation without returning results
    private static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        evaluateAlgorithmWithResults(dataSetA, dataSetB, algorithm, label);
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

                for (int row = 0; row < ENTIRE_BITMAP_SIZE; row++) {
                    try {
                        currentRow.add(Integer.parseInt(values[row].trim()));
                    } catch (NumberFormatException e) {
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
        for (int iteration = 0; iteration < dataSet.size(); iteration++) {
            printRow(iteration, dataSet.get(iteration));
        }
    }

    private static void printLimitedDataSet(List<List<Integer>> dataSet) {
        System.out.println("--- First " + BITMAPS_TO_DISPLAY + " Samples ---");
        for (int iteration = 0; iteration < Math.min(BITMAPS_TO_DISPLAY, dataSet.size()); iteration++) {
            printRow(iteration, dataSet.get(iteration));
        }
    }

    // Prints a single row's pixel values and digit label
    private static void printRow(int digit, List<Integer> row) {
        // The last element is the digit label
        int digitLabel = row.get(BITMAP_SIZE);

        System.out.print("Sample " + (digit + 1) + " (Digit: " + digitLabel + "): [");
        // Print the 64 pixel values
        for (int pixel = 0; pixel < BITMAP_SIZE; pixel++) {
            System.out.print(row.get(pixel));
            if (pixel < BITMAP_SIZE - 1) {
                System.out.print(", ");
            }
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
                    case 1:
                        if (dataSetA != null) {
                            printDataSet(dataSetA);
                        }  
                        break;

                    case 2:
                        if (dataSetB != null) {
                            printDataSet(dataSetB);
                        }   
                        break;

                    case 3:
                        if (dataSetA != null) {
                            printLimitedDataSet(dataSetA);
                        }
                        break;

                    case 4:
                        if (dataSetB != null) {
                            printLimitedDataSet(dataSetB);
                        }
                        break;

                    case 0:
                        running = false;
                        break;

                    default:
                        System.out.println(
                                "\nInvalid choice. Please enter a number corresponding to available actions.");
                }
            } catch (Exception e) {
                System.out.println("\nInvalid input. Please enter a number.");
                scanner.nextLine(); // Consume the invalid input
            }
        }
    }

    private static void UserInterface(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB) {
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
            System.out.println("9 -> Run All Algorithms in Order");
            System.out.println("0 -> Exit");
            System.out.print("Choose bettween 0-9: ");

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
                        // MLP (raw pixels only)
                        evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_ONLY), "MLP [Raw Only]");

                        // MLP with all features
                        evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.ALL), "MLP [All Features]");
                        
                        // MLP with centroid distances
                        evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.CENTROID_ONLY), "MLP [Centroid Only]");
                        
                        // MLP with combined features
                        evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_CENTROID), "MLP [Raw + Centroid]");
                        
                        // MLP with K-Means features
                        evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_KMEANS), "MLP [Raw + KMeans]");
                        
                        // MLP with GA-weighted features
                        evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_GA), "MLP [Raw + GA]");
                        
                        break;

                    case 4:
                        evaluateAlgorithm(dataSetA, dataSetB, DISTANCE_FROM_CENTROID, "Distance From Centroid"); // train on A, test on B
                        break;

                    case 5:
                        // SVM Centroid Distances only
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.CENTROID_ONLY), "SVM [Centroid Only]"); // train on A, test on B

                        // SVM with all features
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.ALL), "SVM [All Features]"); // train on A, test on B

                        // SVM Simple Raw + Centroid mix
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_CENTROID), "SVM [Raw + Centroid]"); // train on A, test on B

                        // SVM Raw + K-Means Distances
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_KMEANS), "SVM [Raw + KMeans]"); // train on A, test on B

                        // SVM Raw + GA Weighted Pixels
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_GA), "SVM [Raw + GA]"); // train on A, test on B
                        break;

                    case 6:
                        evaluateAlgorithm(dataSetA, dataSetB, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour"); // train on A, test on B
                        break;

                    case 7:
                        evaluateAlgorithm(dataSetA, dataSetB, MAHALANOBIS_DISTANCE, "Mahalanobis Distance"); // train on A, test on B
                        break;

                    case 8:
                        evaluateAlgorithm(dataSetA, dataSetB, ALL_AT_ONCE, "All at Once"); // train on A, test on B
                        break;
                    
                    case 9:
                        runAllInOrder(dataSetA, dataSetB);
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

    // Modified runAllInOrder with averages calculation
    private static void runAllInOrder(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB) {
        // Storage for results
        java.util.Map<String, EvaluationResult> resultsAonB = new java.util.LinkedHashMap<>();
        java.util.Map<String, EvaluationResult> resultsBonA = new java.util.LinkedHashMap<>();
        
        System.out.println("\n========================================");
        System.out.println("Running All Algorithms in Sequence");
        System.out.println("Trained on dataset A tested on dataset B");
        System.out.println("========================================\n");
        
        resultsAonB.put("Euclidean Distance", evaluateAlgorithmWithResults(dataSetA, dataSetB, EUCLIDEAN_DISTANCE, "Euclidean Distance"));
        resultsAonB.put("Distance From Centroid", evaluateAlgorithmWithResults(dataSetA, dataSetB, DISTANCE_FROM_CENTROID, "Distance From Centroid"));
        resultsAonB.put("K Nearest Neighbour", evaluateAlgorithmWithResults(dataSetA, dataSetB, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour"));
        resultsAonB.put("Mahalanobis Distance", evaluateAlgorithmWithResults(dataSetA, dataSetB, MAHALANOBIS_DISTANCE, "Mahalanobis Distance"));
        
        AllAtOnce AllsetA = new AllAtOnce();
        AllsetA.predict(dataSetA.get(0), dataSetA);
        
        System.out.println("\n--- Multi-Layer Perceptron Variants ---");
        resultsAonB.put("MLP [Raw Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getMlpRawOnly(), "MLP [Raw Only]"));
        resultsAonB.put("MLP [Centroid Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getMlpCentroidOnly(), "MLP [Centroid Only]"));
        resultsAonB.put("MLP [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getMlpRawCentroid(), "MLP [Raw + Centroid]"));
        resultsAonB.put("MLP [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getMlpRawKMeans(), "MLP [Raw + KMeans]"));
        resultsAonB.put("MLP [Raw + GA]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getMlpRawGA(), "MLP [Raw + GA]"));
        resultsAonB.put("MLP [All Features]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getMlpAll(), "MLP [All Features]"));
        
        System.out.println("\n--- Support Vector Machine Variants ---");
        resultsAonB.put("SVM [Centroid Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getSvmCentroidOnly(), "SVM [Centroid Only]"));
        resultsAonB.put("SVM [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getSvmRawCentroid(), "SVM [Raw + Centroid]"));
        resultsAonB.put("SVM [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getSvmRawKMeans(), "SVM [Raw + KMeans]"));
        resultsAonB.put("SVM [Raw + GA]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getSvmRawGA(), "SVM [Raw + GA]"));
        resultsAonB.put("SVM [All Features]", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA.getSvmAll(), "SVM [All Features]"));
        
        System.out.println("\n--- All Algorithms at once (pick most voted class) ---");
        resultsAonB.put("All at Once", evaluateAlgorithmWithResults(dataSetA, dataSetB, AllsetA, "All at Once"));
        
        // ========== SECOND RUN: B on A ==========
        System.out.println("\n========================================");
        System.out.println("Running All Algorithms in Sequence");
        System.out.println("Trained on dataset B tested on dataset A");
        System.out.println("========================================\n");
        
        resultsBonA.put("Euclidean Distance", evaluateAlgorithmWithResults(dataSetB, dataSetA, EUCLIDEAN_DISTANCE, "Euclidean Distance"));
        resultsBonA.put("Distance From Centroid", evaluateAlgorithmWithResults(dataSetB, dataSetA, DISTANCE_FROM_CENTROID, "Distance From Centroid"));
        resultsBonA.put("K Nearest Neighbour", evaluateAlgorithmWithResults(dataSetB, dataSetA, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour"));
        resultsBonA.put("Mahalanobis Distance", evaluateAlgorithmWithResults(dataSetB, dataSetA, MAHALANOBIS_DISTANCE, "Mahalanobis Distance"));
        
        AllAtOnce AllsetB = new AllAtOnce();
        AllsetB.predict(dataSetB.get(0), dataSetB);
        
        System.out.println("\n--- Multi-Layer Perceptron Variants ---");
        resultsBonA.put("MLP [Raw Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getMlpRawOnly(), "MLP [Raw Only]"));
        resultsBonA.put("MLP [Centroid Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getMlpCentroidOnly(), "MLP [Centroid Only]"));
        resultsBonA.put("MLP [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getMlpRawCentroid(), "MLP [Raw + Centroid]"));
        resultsBonA.put("MLP [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getMlpRawKMeans(), "MLP [Raw + KMeans]"));
        resultsBonA.put("MLP [Raw + GA]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getMlpRawGA(), "MLP [Raw + GA]"));
        resultsBonA.put("MLP [All Features]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getMlpAll(), "MLP [All Features]"));
        
        System.out.println("\n--- Support Vector Machine Variants ---");
        resultsBonA.put("SVM [Centroid Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getSvmCentroidOnly(), "SVM [Centroid Only]"));
        resultsBonA.put("SVM [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getSvmRawCentroid(), "SVM [Raw + Centroid]"));
        resultsBonA.put("SVM [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getSvmRawKMeans(), "SVM [Raw + KMeans]"));
        resultsBonA.put("SVM [Raw + GA]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getSvmRawGA(), "SVM [Raw + GA]"));
        resultsBonA.put("SVM [All Features]", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB.getSvmAll(), "SVM [All Features]"));
        
        System.out.println("\n--- All Algorithms at once (pick most voted class) ---");
        resultsBonA.put("All at Once", evaluateAlgorithmWithResults(dataSetB, dataSetA, AllsetB, "All at Once"));
        
        // ========== CALCULATE AND PRINT AVERAGES ==========
        System.out.println("\n========================================");
        System.out.println("Averages between both runs:");
        System.out.println("========================================\n");
        
        for (String algorithmName : resultsAonB.keySet()) {
            EvaluationResult resultAonB = resultsAonB.get(algorithmName);
            EvaluationResult resultBonA = resultsBonA.get(algorithmName);
            
            System.out.println("--- " + algorithmName + " ---");
            
            if (resultAonB.isSplitResult && resultBonA.isSplitResult) {
                // Handle SVM results
                double avgOneVsRest = (resultAonB.successRateOneVsRest + resultBonA.successRateOneVsRest) / 2.0;
                double avgOneVsOne = (resultAonB.successRateOneVsOne + resultBonA.successRateOneVsOne) / 2.0;
                double avgTime = (resultAonB.evaluationTime + resultBonA.evaluationTime) / 2.0;
                
                System.out.printf("   Average One-vs-Rest Success Rate: %.5f%%%n", avgOneVsRest);
                System.out.printf("   Average One-vs-One Success Rate: %.5f%%%n", avgOneVsOne);
                System.out.printf("   Average Evaluation Time: %.5f seconds%n", avgTime);
            } else {
                // Handle regular results
                double avgSuccessRate = (resultAonB.successRate + resultBonA.successRate) / 2.0;
                double avgTime = (resultAonB.evaluationTime + resultBonA.evaluationTime) / 2.0;
                
                System.out.printf("   A→B Success Rate: %.5f%%%n", resultAonB.successRate);
                System.out.printf("   B→A Success Rate: %.5f%%%n", resultBonA.successRate);
                System.out.printf("   Average Success Rate: %.5f%%%n", avgSuccessRate);
                System.out.printf("   Average Evaluation Time: %.5f seconds%n", avgTime);
            }
            System.out.println();
        }
        
        System.out.println("========================================");
        System.out.println("Evaluation Complete!");
        System.out.println("========================================\n");
    }

    public static void main(String[] args) {
        // read datasets
        List<List<Integer>> dataSetA = readCsvFile(DATASET_A_FILE_PATH);
        List<List<Integer>> dataSetB = readCsvFile(DATASET_B_FILE_PATH);

        // start user interface
        // UserInterface(dataSetA, dataSetB);
        runAllInOrder(dataSetA, dataSetB);
    }
}
