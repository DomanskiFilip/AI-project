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

    // placeholder Algorithm interface
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
            int digitClass = row.get(BITMAP_SIZE); // The class label (0-9)
            countPerClass[digitClass]++;
            for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                sumPerClass[digitClass][featureIndex] += row.get(featureIndex);
            }
        }

        // Calculate the average (centroid)
        for (int digit = 0; digit < 10; digit++) {
            if (countPerClass[digit] > 0) {
                for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                    centroids[digit][featureIndex] = sumPerClass[digit][featureIndex] / countPerClass[digit];
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
        List<Integer> firstSample = trainingSet.get(random.nextInt(trainingSet.size()));
        for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
            centroids[0][featureIndex] = firstSample.get(featureIndex);
        }

        for (int clusterIndex = 1; clusterIndex < clusters; clusterIndex++) {
            double[] distances = new double[trainingSet.size()];
            double totalDistance = 0;
            // Calculate distance to the nearest existing centroid for all samples
            for (int sampleIndex = 0; sampleIndex < trainingSet.size(); sampleIndex++) {
                List<Integer> sample = trainingSet.get(sampleIndex);
                double minDistance = Double.MAX_VALUE;
                for (int existingCluster = 0; existingCluster < clusterIndex; existingCluster++) {
                    double distance = 0;
                    for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                        double diff = sample.get(featureIndex) - centroids[existingCluster][featureIndex];
                        distance += diff * diff;
                    }
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
                distances[sampleIndex] = minDistance;
                totalDistance += minDistance; // Accumulate total distance squared
            }
            // Select new centroid weighted by distance
            double randomValue = random.nextDouble() * totalDistance;
            double cumulativeDistance = 0;
            for (int sampleIndex = 0; sampleIndex < distances.length; sampleIndex++) {
                cumulativeDistance += distances[sampleIndex];
                if (cumulativeDistance >= randomValue) {
                    List<Integer> pickedSample = trainingSet.get(sampleIndex);
                    for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                        centroids[clusterIndex][featureIndex] = pickedSample.get(featureIndex);
                    }
                    break;
                }
            }
        }

        // Iterative refinement (standard K-Means)
        for (int iteration = 0; iteration < KMEANS_MAX_ITERATIONS; iteration++) {
            double[][] sums = new double[clusters][BITMAP_SIZE];
            int[] counts = new int[clusters];

            // Assign points to nearest centroid
            for (List<Integer> sample : trainingSet) {
                int bestCluster = 0;
                double bestDistance = Double.MAX_VALUE;
                for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
                    double distance = 0;
                    for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                        double diff = sample.get(featureIndex) - centroids[clusterIndex][featureIndex];
                        distance += diff * diff;
                    }
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestCluster = clusterIndex;
                    }
                }
                counts[bestCluster]++;
                for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++)
                    sums[bestCluster][featureIndex] += sample.get(featureIndex);
            }

            // Recalculate centroids
            boolean centroidsMoved = false;
            for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
                if (counts[clusterIndex] > 0) {
                    for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                        double newValue = sums[clusterIndex][featureIndex] / counts[clusterIndex];
                        if (Math.abs(newValue - centroids[clusterIndex][featureIndex]) > 1e-6) {
                            centroidsMoved = true;
                        }
                        centroids[clusterIndex][featureIndex] = newValue;
                    }
                }
            }
            if (!centroidsMoved) {
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
        double[] distances = new double[clusters];
        for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
            double sum = 0;
            for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                double diff = sample.get(featureIndex) - kmeansCentroids[clusterIndex][featureIndex];
                sum += diff * diff;
            }
            distances[clusterIndex] = sum;
        }
        return distances;
    }

    private static final int GA_POPULATION = 20;
    private static final int GA_GENERATIONS = 20;
    private static final double GA_MUTATION = 0.05;

    // function to evolve weights using a Genetic Algorithm for feature selection/emphasis
    public static double[] evolveGeneticWeights(List<List<Integer>> trainingSet) {
        Random randomGenerator = new Random(42);
        List<double[]> population = new ArrayList<>();
        // Initialize population (random weights)
        for (int individualIndex = 0; individualIndex < GA_POPULATION; individualIndex++) {
            double[] individual = new double[BITMAP_SIZE];
            for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++)
                individual[featureIndex] = randomGenerator.nextDouble();
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
            for (int individualIndex = 0; individualIndex < population.size(); individualIndex++) {
                fitnessScores[individualIndex] = evaluateWeightFitness(population.get(individualIndex), trainingSet);
                if (fitnessScores[individualIndex] > bestScore) {
                    bestScore = fitnessScores[individualIndex];
                    bestIndividual = population.get(individualIndex);
                    bestIndex = individualIndex;
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
                int parent1Index = randomGenerator.nextInt(population.size());
                int parent2Index = randomGenerator.nextInt(population.size());
                if (parent1Index == bestIndex) {
                    parent1Index = (parent1Index + 1) % population.size();
                }
                if (parent2Index == bestIndex) {
                    parent2Index = (parent2Index + 1) % population.size();
                }
                double[] parent1 = population.get(parent1Index);
                double[] parent2 = population.get(parent2Index);
                // Crossover
                double[] childIndividual = new double[BITMAP_SIZE];
                for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                    childIndividual[featureIndex] = randomGenerator.nextDouble() < 0.5 ? parent1[featureIndex] : parent2[featureIndex];
                }
                    
                // Mutation
                for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                    if (randomGenerator.nextDouble() < GA_MUTATION) {
                         childIndividual[featureIndex] += randomGenerator.nextGaussian() * 0.1;
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
            for (int classIndex = 0; classIndex < centroids.length; classIndex++) {
                double distance = 0;
                for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                    // Apply weight before calculating difference
                    double diff = (sample.get(featureIndex) * weights[featureIndex]) - centroids[classIndex][featureIndex];
                    distance += diff * diff;
                }
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClass = classIndex;
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
        for (int pixelIndex = 0; pixelIndex < BITMAP_SIZE; pixelIndex++) {
             vector[pixelIndex] = sample.get(pixelIndex);
        }
        return vector;
    }

    // Create a feature vector of distances to the 10 class centroids
    public static double[] buildCentroidDistanceVector(List<Integer> sample, double[][] centroids) {
        int numCentroids = (centroids != null) ? centroids.length : 0;
        double[] vector = new double[numCentroids];
        for (int centroidIndex = 0; centroidIndex < numCentroids; centroidIndex++) {
            double sum = 0;
            for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                double diff = sample.get(featureIndex) - centroids[centroidIndex][featureIndex];
                sum += diff * diff;
            }
            vector[centroidIndex] = Math.sqrt(sum);
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
        for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
            double sum = 0;
            for (int featureIndex = 0; featureIndex < BITMAP_SIZE; featureIndex++) {
                double diff = sample.get(featureIndex) - kmeansCentroids[clusterIndex][featureIndex];
                sum += diff * diff;
            }
            vector[clusterIndex] = Math.sqrt(sum);
        }
        return vector;
    }

    // Create a feature vector of raw pixels multiplied by the Genetic Algorithm weights
    public static double[] buildGAWeightedVector(List<Integer> sample, double[] gaWeights) {
        double[] vector = new double[BITMAP_SIZE];
        for (int pixelIndex = 0; pixelIndex < BITMAP_SIZE; pixelIndex++) {
            vector[pixelIndex] = sample.get(pixelIndex) * (gaWeights != null ? gaWeights[pixelIndex] : 1.0);
        }
        return vector;
    }

    // Utility function to combine multiple feature vectors into one long vector
    public static double[] concatVectors(double[]... parts) {
        int totalLength = 0;
        for (double[] part : parts) {
            if (part != null) {
                totalLength += part.length;
            }
        }
        double[] output = new double[totalLength];
        int position = 0;
        for (double[] part : parts) {
            if (part == null) {
                continue;
            }
            System.arraycopy(part, 0, output, position, part.length);
            position += part.length;
        }
        return output;
    }

    // Combination of all feature vectors used in the 'ALL' mode of SVM
    private static double[] buildCombinedFeatureVector(List<Integer> sample, double[][] centroidCache, double[][] kmeansCache, double[] gaWeightsCache) {
        // Raw pixels
        double[] rawPixels = buildRawPixelsVector(sample);
        // Distances to class centroids
        double[] centroidDistances = buildCentroidDistanceVector(sample, centroidCache);
        // Distances to K-Means centroids
        double[] kmeansDistances = buildKMeansDistanceVector(sample, kmeansCache);
        // GA-weighted raw pixels
        double[] gaWeighted = buildGAWeightedVector(sample, gaWeightsCache);

        // Concatenate all of them
        return concatVectors(rawPixels, centroidDistances, kmeansDistances, gaWeighted);
    }

    // ------------------------------------------------------------------------
    // AI ALGORITHMS:
    // ------------------------------------------------------------------------

    // Euclidean Distance Algorithm
    private static class EuclideanDistance implements Algorithm {

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            double minDistance = Double.MAX_VALUE;
            List<Integer> closest = null;
            for (List<Integer> candidate : trainingSet) {
                double sum = 0;
                for (int pixelIndex = 0; pixelIndex < BITMAP_SIZE; pixelIndex++) {
                    double distance = sample.get(pixelIndex) - candidate.get(pixelIndex);
                    sum += distance * distance;
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
        private static final int PERCEPTRONS = 300;
        private static final int EPOCHS = 50;
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
            
            double[] inputs = buildFeatureVector(sample); // create all bitmaps with selected features

            // normalize bitmaps with exclusion of original raw bitmap
            if (mode != FeatureMode.RAW_ONLY){
                inputs = normalizeFeatures(inputs);
            }
            
            double[] outputs = forward(inputs); // forward pass
            
            int bestIndex = 0;
            double bestValue = outputs[0];
            for (int outputIndex = 1; outputIndex < outputs.length; outputIndex++) {
                if (outputs[outputIndex] > bestValue) {
                    bestValue = outputs[outputIndex];
                    bestIndex = outputIndex;
                }
            }
            return Integer.valueOf(bestIndex);
        }

        private void train(List<List<Integer>> trainingSet) {
            // Compute feature caches based on mode
            computeFeatureCaches(trainingSet);
            
            // Determine input size from feature vector
            inputSize = buildFeatureVector(trainingSet.get(0)).length;
            
            // compute normalization of bitmaps with exclusion of original raw bitmap
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
                    for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                        double sum = biasHidden[hiddenIndex];
                        for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                            sum += weightsInputHidden[hiddenIndex][inputIndex] * inputs[inputIndex];
                        }
                        hidden[hiddenIndex] = sigmoid(sum);
                    }

                    // Forward pass - output layer
                    for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                        double sum = biasOutput[outputIndex];
                        for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                            sum += weightsHiddenOutput[outputIndex][hiddenIndex] * hidden[hiddenIndex];
                        }
                        outputs[outputIndex] = sigmoid(sum);
                    }

                    // Prepare target vector
                    double[] target = new double[CLASSES];
                    target[targetClass] = 1.0;

                    // Backpropagation - output layer
                    double[] outputDeltas = new double[CLASSES];
                    for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                        double error = target[outputIndex] - outputs[outputIndex];
                        outputDeltas[outputIndex] = error * sigmoidDerivative(outputs[outputIndex]);
                    }

                    // Backpropagation - hidden layer
                    double[] hiddenDeltas = new double[PERCEPTRONS];
                    for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                        double error = 0;
                        for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                            error += outputDeltas[outputIndex] * weightsHiddenOutput[outputIndex][hiddenIndex];
                        }
                        hiddenDeltas[hiddenIndex] = error * sigmoidDerivative(hidden[hiddenIndex]);
                    }

                    // Update weights - input to hidden
                    for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                        for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                            weightsInputHidden[hiddenIndex][inputIndex] += LEARNING_RATE * hiddenDeltas[hiddenIndex] * inputs[inputIndex];
                        }
                        biasHidden[hiddenIndex] += LEARNING_RATE * hiddenDeltas[hiddenIndex];
                    }

                    // Update weights - hidden to output
                    for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                        for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                            weightsHiddenOutput[outputIndex][hiddenIndex] += LEARNING_RATE * outputDeltas[outputIndex] * hidden[hiddenIndex];
                        }
                        biasOutput[outputIndex] += LEARNING_RATE * outputDeltas[outputIndex];
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
            double[] rawPixels = buildRawPixelsVector(sample);

            switch (mode) {
                case RAW_ONLY:
                    return rawPixels;
                case CENTROID_ONLY:
                    return buildCentroidDistanceVector(sample, centroidCache);
                case RAW_CENTROID:
                    return concatVectors(rawPixels, buildCentroidDistanceVector(sample, centroidCache));
                case RAW_KMEANS:
                    return concatVectors(rawPixels, buildKMeansDistanceVector(sample, kmeansCentroidCache));
                case RAW_GA:
                    return concatVectors(rawPixels, buildGAWeightedVector(sample, gaWeightsCache));
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
                for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
                    featureMean[featureIndex] += features[featureIndex];
                }
            }
            for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
                featureMean[featureIndex] /= trainingSet.size();
            }
            
            // Compute standard deviation
            for (List<Integer> row : trainingSet) {
                double[] features = buildFeatureVector(row);
                for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
                    double diff = features[featureIndex] - featureMean[featureIndex];
                    featureStd[featureIndex] += diff * diff;
                }
            }
            for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
                featureStd[featureIndex] = Math.sqrt(featureStd[featureIndex] / trainingSet.size());
                // Prevent division by zero
                if (featureStd[featureIndex] < 1e-9) {
                    featureStd[featureIndex] = 1.0;
                }
            }
        }

        // Normalize features using computed statistics
        private double[] normalizeFeatures(double[] features) {
            double[] normalized = new double[features.length];
            for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
                normalized[featureIndex] = (features[featureIndex] - featureMean[featureIndex]) / featureStd[featureIndex];
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
            for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                biasHidden[hiddenIndex] = (random.nextDouble() * 2 - 1) * range;
                for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                    weightsInputHidden[hiddenIndex][inputIndex] = (random.nextDouble() * 2 - 1) * range;
                }
            }
            for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                biasOutput[outputIndex] = (random.nextDouble() * 2 - 1) * range;
                for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                    weightsHiddenOutput[outputIndex][hiddenIndex] = (random.nextDouble() * 2 - 1) * range;
                }
            }
        }

        private double[] forward(double[] inputs) {
            double[] hidden = new double[PERCEPTRONS];
            for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                double sum = biasHidden[hiddenIndex];
                for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                    sum += weightsInputHidden[hiddenIndex][inputIndex] * inputs[inputIndex];
                }
                hidden[hiddenIndex] = sigmoid(sum);
            }

            double[] outputs = new double[CLASSES];
            for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                double sum = biasOutput[outputIndex];
                for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                    sum += weightsHiddenOutput[outputIndex][hiddenIndex] * hidden[hiddenIndex];
                }
                outputs[outputIndex] = sigmoid(sum);
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
            return "Mode: " + mode + ", Epochs: " + EPOCHS + ", Learning rate: " + LEARNING_RATE + ", Perceptrons: " + PERCEPTRONS +  ", Random seed: " + RANDOM_SEED;
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
                for (int pixelIndex = 0; pixelIndex < BITMAP_SIZE; pixelIndex++) {
                    double diff = sample.get(pixelIndex) - centroids[digit][pixelIndex];
                    sum += diff * diff;
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

            int numSamples = trainingSet.size();
            // Projects all samples to the feature space to build the training matrix
            double[] firstSample = projectToFeatureSpace(trainingSet.get(0));
            int featureSize = firstSample.length;

            double[][] featureMatrix = new double[numSamples][featureSize];
            int[] labels = new int[numSamples];

            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                List<Integer> row = trainingSet.get(sampleIndex);
                featureMatrix[sampleIndex] = projectToFeatureSpace(row);
                labels[sampleIndex] = row.get(BITMAP_SIZE);
            }

            // calculate mean/standard deviation and scale the training data for normalization purposes 
            // Xnormalized = (X - mean) / standard deviation
            FeatureNormalizer normalizer = new FeatureNormalizer(featureSize);
            normalizer.fitAndTransform(featureMatrix); // Scales the training data in place

            // Train One-vs-Rest (10 classifiers vs all others)
            this.oneVsRestModel = trainPerceptronModel(featureMatrix, labels, normalizer);

            // Train One-vs-One (45 binary classifiers: 0 vs 1, 0 vs 2, ..., 8 vs 9)
            this.oneVsOneModels = new ArrayList<>();
            for (int classA = 0; classA < CLASSES; classA++) {
                for (int classB = classA + 1; classB < CLASSES; classB++) {
                    List<Integer> pairIndices = new ArrayList<>();
                    for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                        if (labels[sampleIndex] == classA || labels[sampleIndex] == classB) {
                            pairIndices.add(sampleIndex);
                        }
                    }

                    if (!pairIndices.isEmpty()) {
                        // Train a binary classifier for this specific pair
                        Model pairModel = trainLinearPerceptron(featureMatrix, labels, pairIndices, classA, classB);
                        oneVsOneModels.add(pairModel);
                    }
                }
            }
        }

        // maps the input sample to the selected bitmap / feature vector size
        private double[] projectToFeatureSpace(List<Integer> sample) {
            double[] rawPixels = buildRawPixelsVector(sample);

            switch (mode) {
                case CENTROID_ONLY:
                    return buildCentroidDistanceVector(sample, centroidCache);
                case RAW_CENTROID:
                    return concatVectors(rawPixels, buildCentroidDistanceVector(sample, centroidCache));
                case RAW_KMEANS:
                    return concatVectors(rawPixels, buildKMeansDistanceVector(sample, kmeansCentroidCache));
                case RAW_GA:
                    return concatVectors(rawPixels, buildGAWeightedVector(sample, gaWeightsCache));
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
            for (int classIndex = 0; classIndex < CLASSES; classIndex++) {
                if (votes[classIndex] > maxVotes) {
                    maxVotes = votes[classIndex];
                    bestClass = classIndex;
                }
            }
            return bestClass;
        }

        // Training for the Multiclass One-vs-Rest model
        private Model trainPerceptronModel(double[][] features, int[] labels, FeatureNormalizer normalizer) {
            int featureSize = features[0].length;
            Model model = new Model(CLASSES, featureSize, normalizer);
            int numSamples = features.length;
            List<Integer> indices = new ArrayList<>();
            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                indices.add(sampleIndex);
            }
            Random rand = new Random(RANDOM_SEED);

            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                Collections.shuffle(indices, rand); // Shuffle data for stochastic training (shuffling the dataset to avoid bias)
                int mistakes = 0;
                for (int idx : indices) {
                    double[] sample = features[idx];
                    int target = labels[idx];
                    PredictionResult res = model.computeScores(sample);

                    // Find the best scoring rival class (the one predicted incorrectly)
                    int bestRival = -1;
                    double bestRivalScore = Double.NEGATIVE_INFINITY;
                    for (int classIndex = 0; classIndex < CLASSES; classIndex++) {
                        if (classIndex == target) {
                            continue;
                        }
                        if (res.scores[classIndex] > bestRivalScore) {
                            bestRivalScore = res.scores[classIndex];
                            bestRival = classIndex;
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
                    int targetLabel = (labels[idx] == classA) ? 1 : -1;
                    double score = model.pointInFeatureSpace(0, features[idx]) + model.bias[0];

                    // Perceptron learning rule
                    if (targetLabel * score <= MARGIN) {
                        for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
                            model.weights[0][featureIndex] += LEARNING_RATE * targetLabel * features[idx][featureIndex];
                        }
                        model.bias[0] += LEARNING_RATE * targetLabel;
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
                double maxScore = Double.NEGATIVE_INFINITY;
                for (int classIndex = 0; classIndex < weights.length; classIndex++) {
                    scores[classIndex] = pointInFeatureSpace(classIndex, features) + bias[classIndex];
                    if (scores[classIndex] > maxScore) {
                        maxScore = scores[classIndex];
                        predicted = classIndex;
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

                for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
                    sum += weights[classIdx][featureIndex] * features[featureIndex];
                }

                return sum;
            }

            // Applies the learning update to the weights and bias for a given class
            void update(double[] features, int classIdx, double direction) {
                for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
                    weights[classIdx][featureIndex] += LEARNING_RATE * direction * features[featureIndex];
                }
                bias[classIdx] += LEARNING_RATE * direction;
            }

            // Adds current weights/biases to the running sums for averaging
            void accumulateAverages() {
                steps++;
                for (int classIndex = 0; classIndex < weights.length; classIndex++) {
                    for (int featureIndex = 0; featureIndex < weights[0].length; featureIndex++) {
                        weightSums[classIndex][featureIndex] += weights[classIndex][featureIndex];
                    }
                    biasSums[classIndex] += bias[classIndex];
                }
            }

            // Calculates the final averaged weights
            void finalizeWeights() {
                if (steps == 0) {
                    return;
                }
                for (int classIndex = 0; classIndex < weights.length; classIndex++) {
                    for (int featureIndex = 0; featureIndex < weights[0].length; featureIndex++) {
                        weights[classIndex][featureIndex] = weightSums[classIndex][featureIndex] / steps;
                    }
                    bias[classIndex] = biasSums[classIndex] / steps;
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
                for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
                    double sum = 0, sumSquared = 0;
                    for (double[] row : matrix) {
                        sum += row[featureIndex];
                        sumSquared += row[featureIndex] * row[featureIndex];
                    }
                    mean[featureIndex] = sum / matrix.length;
                    double variance = (sumSquared / matrix.length) - (mean[featureIndex] * mean[featureIndex]);
                    // Use a small epsilon to prevent division by zero for constant features
                    std[featureIndex] = Math.max(Math.sqrt(Math.max(variance, 0)), 1e-9);
                }

                // Apply Normalization (Transform)
                for (double[] row : matrix) {
                    for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
                        row[featureIndex] = (row[featureIndex] - mean[featureIndex]) / std[featureIndex];
                    }
                }
            }

            // Applies normalization to a single input vector using pre-calculated stats
            double[] normalize(double[] vector) {
                double[] output = new double[featureSize];
                for (int featureIndex = 0; featureIndex < featureSize; featureIndex++)
                    output[featureIndex] = (vector[featureIndex] - mean[featureIndex]) / std[featureIndex];
                return output;
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
                for (int pixelIndex = 0; pixelIndex < BITMAP_SIZE; pixelIndex++) {
                    double diff = sample.get(pixelIndex) - candidate.get(pixelIndex);
                    distance += diff * diff;
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

    // Mahalanobis Distance Algorithm with Per-Class Covariance
    private static class MahalanobisDistance implements Algorithm {
        private static final int CLASSES = 10;

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            int featureCount = BITMAP_SIZE; // Only using the 64 pixels as features

            // Pre-calculations needed for Mahalanobis Distance
            double[][] centroids = calculateCentroids(trainingSet);
            int[] classCounts = new int[CLASSES];
            
            // Separate training data by class
            List<List<List<Integer>>> classSamples = new ArrayList<>();
            for (int i = 0; i < CLASSES; i++) {
                classSamples.add(new ArrayList<>());
            }
            
            for (List<Integer> row : trainingSet) {
                int label = row.get(BITMAP_SIZE);
                classCounts[label]++;
                classSamples.get(label).add(row);
            }

            // Calculate per-class covariance matrices and their inverses
            double[][][] inverseCovariances = new double[CLASSES][][];
            for (int digit = 0; digit < CLASSES; digit++) {
                if (classCounts[digit] > featureCount) { // Need enough samples
                    double[][] covariance = computeClassCovarianceMatrix(
                        classSamples.get(digit), 
                        featureCount, 
                        centroids[digit]
                    );
                    inverseCovariances[digit] = invertMatrix(covariance);
                } else {
                    inverseCovariances[digit] = null; // Fallback to Euclidean for this class
                }
            }

            double[] sampleVector = new double[featureCount];
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                sampleVector[featureIndex] = sample.get(featureIndex);
            }

            // Calculate distance to each class centroid
            double bestDistance = Double.MAX_VALUE;
            int bestClass = -1;
            for (int digit = 0; digit < CLASSES; digit++) {
                if (classCounts[digit] == 0) {
                    continue;
                }

                // Difference vector: Sample - Centroid
                double[] diff = new double[featureCount];
                for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                    diff[featureIndex] = sampleVector[featureIndex] - centroids[digit][featureIndex];
                }

                double distance;
                if (inverseCovariances[digit] != null) {
                    // Use Mahalanobis Distance with per-class covariance
                    distance = computeMahalanobisDistance(diff, inverseCovariances[digit]);
                } else {
                    // Fallback to Euclidean distance if covariance couldn't be computed
                    distance = computeEuclideanDistance(diff);
                }
                
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClass = digit;
                }
            }
            return Integer.valueOf(bestClass >= 0 ? bestClass : 0);
        }

        // Calculates the Covariance Matrix for a single class
        private static double[][] computeClassCovarianceMatrix(
                List<List<Integer>> classSamples, 
                int featureCount, 
                double[] classCentroid) {
            
            double[][] covariance = new double[featureCount][featureCount];
            if (classSamples.size() <= 1) {
                // Not enough samples - return identity-like matrix
                for (int i = 0; i < featureCount; i++) {
                    covariance[i][i] = 1.0;
                }
                return covariance;
            }

            for (List<Integer> row : classSamples) {
                for (int rowFeatureIndex = 0; rowFeatureIndex < featureCount; rowFeatureIndex++) {
                    double diffRow = row.get(rowFeatureIndex) - classCentroid[rowFeatureIndex];
                    for (int colFeatureIndex = 0; colFeatureIndex < featureCount; colFeatureIndex++) {
                        double diffCol = row.get(colFeatureIndex) - classCentroid[colFeatureIndex];
                        covariance[rowFeatureIndex][colFeatureIndex] += diffRow * diffCol;
                    }
                }
            }

            // Normalize and add regularization to the diagonal
            double denominator = classSamples.size() - 1.0;
            for (int rowIndex = 0; rowIndex < featureCount; rowIndex++) {
                for (int colIndex = 0; colIndex < featureCount; colIndex++) {
                    covariance[rowIndex][colIndex] /= denominator;
                }
                // Increased regularization for stability with limited samples
                covariance[rowIndex][rowIndex] += 0.1;
            }
            return covariance;
        }

        // Fallback Euclidean distance calculation
        private static double computeEuclideanDistance(double[] diff) {
            double sumSquares = 0;
            for (double d : diff) {
                sumSquares += d * d;
            }
            return Math.sqrt(sumSquares);
        }

        // Uses Gauss-Jordan elimination to compute the inverse of a matrix
        private static double[][] invertMatrix(double[][] matrix) {
            int size = matrix.length;
            double[][] augmented = new double[size][2 * size];

            // Build augmented matrix [A | I]
            for (int rowIndex = 0; rowIndex < size; rowIndex++) {
                System.arraycopy(matrix[rowIndex], 0, augmented[rowIndex], 0, size);
                augmented[rowIndex][rowIndex + size] = 1.0;
            }

            // Apply row operations to transform [A | I] into [I | A^-1]
            for (int colIndex = 0; colIndex < size; colIndex++) {
                // Find pivot
                int pivotRow = colIndex;
                double maxValue = Math.abs(augmented[pivotRow][colIndex]);
                for (int rowIndex = colIndex + 1; rowIndex < size; rowIndex++) {
                    double value = Math.abs(augmented[rowIndex][colIndex]);
                    if (value > maxValue) {
                        maxValue = value;
                        pivotRow = rowIndex;
                    }
                }

                if (Math.abs(augmented[pivotRow][colIndex]) < 1e-9) {
                    return null; // Matrix is singular
                }

                // Swap rows
                if (pivotRow != colIndex) {
                    double[] temp = augmented[pivotRow];
                    augmented[pivotRow] = augmented[colIndex];
                    augmented[colIndex] = temp;
                }

                // Normalize pivot row
                double pivotValue = augmented[colIndex][colIndex];
                for (int elementIndex = 0; elementIndex < 2 * size; elementIndex++) {
                    augmented[colIndex][elementIndex] /= pivotValue;
                }

                // Eliminate other entries in the column
                for (int rowIndex = 0; rowIndex < size; rowIndex++) {
                    if (rowIndex == colIndex) {
                        continue;
                    }
                    double factor = augmented[rowIndex][colIndex];
                    for (int elementIndex = 0; elementIndex < 2 * size; elementIndex++) {
                        augmented[rowIndex][elementIndex] -= factor * augmented[colIndex][elementIndex];
                    }
                }
            }

            // Extract the inverse matrix A^-1 from the right side
            double[][] inverse = new double[size][size];
            for (int rowIndex = 0; rowIndex < size; rowIndex++) {
                System.arraycopy(augmented[rowIndex], size, inverse[rowIndex], 0, size);
            }
            return inverse;
        }

        // Calculates the Mahalanobis Distance for a given difference vector and inverse covariance
        private static double computeMahalanobisDistance(double[] diff, double[][] inverseCovariance) {
            // intermediate = Covariance^-1 * diff
            double[] intermediate = new double[diff.length];
            for (int rowIndex = 0; rowIndex < diff.length; rowIndex++) {
                double sum = 0;
                for (int colIndex = 0; colIndex < diff.length; colIndex++) {
                    sum += inverseCovariance[rowIndex][colIndex] * diff[colIndex];
                }
                intermediate[rowIndex] = sum;
            }
            
            // distance = diff^T * intermediate
            double distance = 0;
            for (int index = 0; index < diff.length; index++) {
                distance += diff[index] * intermediate[index];
            }
            return Math.sqrt(Math.max(0, distance)); // Protect against numerical errors
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
                int[] predictions = (int[]) result;
                if (predictions[0] == actualDigit) {
                    correctOneVsRest++;
                }
                if (predictions[1] == actualDigit) {
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

                for (int columnIndex = 0; columnIndex < ENTIRE_BITMAP_SIZE; columnIndex++) {
                    try {
                        currentRow.add(Integer.parseInt(values[columnIndex].trim()));
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
        for (int sampleIndex = 0; sampleIndex < dataSet.size(); sampleIndex++) {
            printRow(sampleIndex, dataSet.get(sampleIndex));
        }
    }

    private static void printLimitedDataSet(List<List<Integer>> dataSet) {
        System.out.println("--- First " + BITMAPS_TO_DISPLAY + " Samples ---");
        for (int sampleIndex = 0; sampleIndex < Math.min(BITMAPS_TO_DISPLAY, dataSet.size()); sampleIndex++) {
            printRow(sampleIndex, dataSet.get(sampleIndex));
        }
    }

    // Prints a single row's pixel values and digit label
    private static void printRow(int rowNumber, List<Integer> row) {
        // The last element is the digit label
        int digitLabel = row.get(BITMAP_SIZE);

        System.out.print("Sample " + (rowNumber + 1) + " (Digit: " + digitLabel + "): [");
        // Print the 64 pixel values
        for (int pixelIndex = 0; pixelIndex < BITMAP_SIZE; pixelIndex++) {
            System.out.print(row.get(pixelIndex));
            if (pixelIndex < BITMAP_SIZE - 1) {
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
            System.out.print("Choose between 0-4: ");
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
            System.out.print("Choose between 0-9: ");

            try {
                int choice = scanner.nextInt();
                switch (choice) {
                    case 1:
                        PrintDataUserInterface(dataSetA, dataSetB, scanner);
                        break;

                    case 2:
                        System.out.println("Trained on A tested on B:");
                        evaluateAlgorithm(dataSetA, dataSetB, EUCLIDEAN_DISTANCE, "Euclidean Distance");
                        System.out.println("Trained on B tested on A:");
                        evaluateAlgorithm(dataSetB, dataSetA, EUCLIDEAN_DISTANCE, "Euclidean Distance");
                        break;

                    case 3:
                        System.out.println("Trained on A tested on B:");
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

                        System.out.println("Trained on B tested on A:");

                        // MLP (raw pixels only)
                        evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_ONLY), "MLP [Raw Only]");

                        // MLP with all features
                        evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.ALL), "MLP [All Features]");
                        
                        // MLP with centroid distances
                        evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.CENTROID_ONLY), "MLP [Centroid Only]");
                        
                        // MLP with combined features
                        evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_CENTROID), "MLP [Raw + Centroid]");
                        
                        // MLP with K-Means features
                        evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_KMEANS), "MLP [Raw + KMeans]");
                        
                        // MLP with GA-weighted features
                        evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(MultiLayerPerceptron.FeatureMode.RAW_GA), "MLP [Raw + GA]");
                        
                        break;

                    case 4:
                        System.out.println("Trained on A tested on B:");
                        evaluateAlgorithm(dataSetA, dataSetB, DISTANCE_FROM_CENTROID, "Distance From Centroid");
                        System.out.println("Trained on B tested on A:");
                        evaluateAlgorithm(dataSetB, dataSetA, DISTANCE_FROM_CENTROID, "Distance From Centroid");
                        break;

                    case 5:
                        System.out.println("Trained on A tested on B:");
                        // SVM Centroid Distances only
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.CENTROID_ONLY), "SVM [Centroid Only]");

                        // SVM with all features
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.ALL), "SVM [All Features]");

                        // SVM Simple Raw + Centroid mix
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_CENTROID), "SVM [Raw + Centroid]"); 

                        // SVM Raw + K-Means Distances
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_KMEANS), "SVM [Raw + KMeans]"); 

                        // SVM Raw + GA Weighted Pixels
                        evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_GA), "SVM [Raw + GA]");
                        System.out.println("Trained on B tested on A:");
                        // SVM Centroid Distances only
                        evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(SupportVectorMachine.FeatureMode.CENTROID_ONLY), "SVM [Centroid Only]");

                        // SVM with all features
                        evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(SupportVectorMachine.FeatureMode.ALL), "SVM [All Features]");

                        // SVM Simple Raw + Centroid mix
                        evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_CENTROID), "SVM [Raw + Centroid]"); 

                        // SVM Raw + K-Means Distances
                        evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_KMEANS), "SVM [Raw + KMeans]"); 

                        // SVM Raw + GA Weighted Pixels
                        evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(SupportVectorMachine.FeatureMode.RAW_GA), "SVM [Raw + GA]");

                        break;

                    case 6:
                        System.out.println("Trained on A tested on B:");
                        evaluateAlgorithm(dataSetA, dataSetB, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour");
                        System.out.println("Trained on B tested on A:");
                        evaluateAlgorithm(dataSetB, dataSetA, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour");
                        break;

                    case 7:
                        System.out.println("Trained on A tested on B:");
                        evaluateAlgorithm(dataSetA, dataSetB, MAHALANOBIS_DISTANCE, "Mahalanobis Distance");
                        System.out.println("Trained on B tested on A:");
                        evaluateAlgorithm(dataSetB, dataSetA, MAHALANOBIS_DISTANCE, "Mahalanobis Distance");
                        break;

                    case 8:
                        System.out.println("Trained on A tested on B:");
                        evaluateAlgorithm(dataSetA, dataSetB, ALL_AT_ONCE, "All at Once");
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
        
        AllAtOnce allsetA = new AllAtOnce();
        allsetA.predict(dataSetA.get(0), dataSetA);
        
        System.out.println("\n--- Multi-Layer Perceptron Variants ---");
        resultsAonB.put("MLP [Raw Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawOnly(), "MLP [Raw Only]"));
        resultsAonB.put("MLP [Centroid Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpCentroidOnly(), "MLP [Centroid Only]"));
        resultsAonB.put("MLP [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawCentroid(), "MLP [Raw + Centroid]"));
        resultsAonB.put("MLP [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawKMeans(), "MLP [Raw + KMeans]"));
        resultsAonB.put("MLP [Raw + GA]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawGA(), "MLP [Raw + GA]"));
        resultsAonB.put("MLP [All Features]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpAll(), "MLP [All Features]"));
        
        System.out.println("\n--- Support Vector Machine Variants ---");
        resultsAonB.put("SVM [Centroid Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmCentroidOnly(), "SVM [Centroid Only]"));
        resultsAonB.put("SVM [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmRawCentroid(), "SVM [Raw + Centroid]"));
        resultsAonB.put("SVM [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmRawKMeans(), "SVM [Raw + KMeans]"));
        resultsAonB.put("SVM [Raw + GA]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmRawGA(), "SVM [Raw + GA]"));
        resultsAonB.put("SVM [All Features]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmAll(), "SVM [All Features]"));
        
        System.out.println("\n--- All Algorithms at once (pick most voted class) ---");
        resultsAonB.put("All at Once", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA, "All at Once"));
        
        // ========== SECOND RUN: B on A ==========
        System.out.println("\n========================================");
        System.out.println("Running All Algorithms in Sequence");
        System.out.println("Trained on dataset B tested on dataset A");
        System.out.println("========================================\n");
        
        resultsBonA.put("Euclidean Distance", evaluateAlgorithmWithResults(dataSetB, dataSetA, EUCLIDEAN_DISTANCE, "Euclidean Distance"));
        resultsBonA.put("Distance From Centroid", evaluateAlgorithmWithResults(dataSetB, dataSetA, DISTANCE_FROM_CENTROID, "Distance From Centroid"));
        resultsBonA.put("K Nearest Neighbour", evaluateAlgorithmWithResults(dataSetB, dataSetA, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour"));
        resultsBonA.put("Mahalanobis Distance", evaluateAlgorithmWithResults(dataSetB, dataSetA, MAHALANOBIS_DISTANCE, "Mahalanobis Distance"));
        
        AllAtOnce allsetB = new AllAtOnce();
        allsetB.predict(dataSetB.get(0), dataSetB);
        
        System.out.println("\n--- Multi-Layer Perceptron Variants ---");
        resultsBonA.put("MLP [Raw Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawOnly(), "MLP [Raw Only]"));
        resultsBonA.put("MLP [Centroid Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpCentroidOnly(), "MLP [Centroid Only]"));
        resultsBonA.put("MLP [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawCentroid(), "MLP [Raw + Centroid]"));
        resultsBonA.put("MLP [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawKMeans(), "MLP [Raw + KMeans]"));
        resultsBonA.put("MLP [Raw + GA]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawGA(), "MLP [Raw + GA]"));
        resultsBonA.put("MLP [All Features]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpAll(), "MLP [All Features]"));
        
        System.out.println("\n--- Support Vector Machine Variants ---");
        resultsBonA.put("SVM [Centroid Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmCentroidOnly(), "SVM [Centroid Only]"));
        resultsBonA.put("SVM [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmRawCentroid(), "SVM [Raw + Centroid]"));
        resultsBonA.put("SVM [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmRawKMeans(), "SVM [Raw + KMeans]"));
        resultsBonA.put("SVM [Raw + GA]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmRawGA(), "SVM [Raw + GA]"));
        resultsBonA.put("SVM [All Features]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmAll(), "SVM [All Features]"));
        
        System.out.println("\n--- All Algorithms at once (pick most voted class) ---");
        resultsBonA.put("All at Once", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB, "All at Once"));
        
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
                
                System.out.printf("   A>B Success Rate: %.5f%%%n", resultAonB.successRate);
                System.out.printf("   B>A Success Rate: %.5f%%%n", resultBonA.successRate);
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
         UserInterface(dataSetA, dataSetB);
        // runAllInOrder(dataSetA, dataSetB);
    }
}