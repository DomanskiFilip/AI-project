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
        Object predict(List<Integer> sample, List<List<Integer>> trainingSet); // sample -> row to predict, trainingSet -> dataset A (usually)
    }

    private static final Algorithm EUCLIDEAN_DISTANCE = new EuclideanDistance();
    private static final Algorithm MULTI_LAYER_PERCEPTRON = new MultiLayerPerceptron();
    private static final Algorithm DISTANCE_FROM_CENTROID = new DistanceFromCentroid();
    private static final Algorithm SUPPORT_VECTOR_MACHINE = new SupportVectorMachine();
    private static final Algorithm K_NEAREST_NEIGHBOUR = new K_NEAREST_NEIGHBOUR();
    private static final Algorithm K_MEANS = new KMeans();
    private static final Algorithm MAHALANOBIS_DISTANCE = new MahalanobisDistance();
    private static final Algorithm GENETIC_ALGORYTHM = new GeneticAlgorythm();
    private static final Algorithm ALL_AT_ONCE = new AllAtOnce();

    // Euclidean Distance Algorythm
    private static class EuclideanDistance implements Algorithm {

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (trainingSet == null || trainingSet.isEmpty()) {
                throw new IllegalArgumentException("Training dataset must not be null or empty.");
            }

            double minDistance = Double.MAX_VALUE;
            List<Integer> closest = null;

            for (int c = 0; c < trainingSet.size(); c++) {
                List<Integer> candidate = trainingSet.get(c);
                double sum = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double distance = sample.get(i) - candidate.get(i);
                    sum += distance * distance; // squared distance gives better results than square root of distance
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
        // modify these parameters to tune the neural network:
        private static final int PERCEPTRONS = 500; // number of neurons in the hidden layer
        private static final int EPOCHS = 500; // number of training iterations
        private static final double LEARNING_RATE = 0.1; // learning rate for weight update
        private static final long RANDOM_SEED = 42; // Fixed seed for the random number generator to ensure reproducibility.
        // This guarantees that the weights and biases are initialized to the same starting values every time the program is run, leading to consistent results


        // fixed parameters:
        private static final int CLASSES = 10; // number of classes (digits 0-9)
        private double[][] weightsInputHidden;  // weights between input and hidden layer [perceptron][features]
        private double[] biasHidden; // biases for hidden layer
        private double[][] weightsHiddenOutput; // weights between hidden and output layer [classes][perceptron]
        private double[] biasOutput; // biases for output layer
        private boolean trained = false;

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (!trained) {
                train(trainingSet);  // Train the network on the trainingSet set (dataset A)
            }
            double[] inputs = toInputVector(sample);
            double[] outputs = forward(inputs);
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
            if (trainingSet == null || trainingSet.isEmpty()) {
                throw new IllegalArgumentException("Training set is empty.");
            }
            initializeWeights();

            // loop running epochs of training
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                for (List<Integer> row : trainingSet) {
                    double[] inputs = toInputVector(row);
                    int targetClass = row.get(BITMAP_SIZE);
                    double[] hidden = new double[PERCEPTRONS];
                    double[] outputs = new double[CLASSES];

                    // forward pass to calculate hidden layer activations
                    for (int h = 0; h < PERCEPTRONS; h++) {
                        double sum = biasHidden[h];
                        for (int i = 0; i < BITMAP_SIZE; i++) {
                            sum += weightsInputHidden[h][i] * inputs[i];
                        }
                        hidden[h] = sigmoid(sum);
                    }

                    // forward pass to calculate output layer activations
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
        
        // method to read inputed values representing the digit from sample into input vector from int -> double
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

    // helper function for calculateing centroids of each class for SVM and Distance from Centroid algorythms
    public static double[][] calculateCentroids(List<List<Integer>> trainingSet) {
        double[][] centroids = new double[10][BITMAP_SIZE];

        // Initialize arrays to store sum and count for each class
        double[][] sumPerClass = new double[10][BITMAP_SIZE];
        int[] countPerClass = new int[10];

        // Sum all dimensions for each digit/class
        for (List<Integer> row : trainingSet) {
            int instance = row.get(BITMAP_SIZE);
            countPerClass[instance]++;
            for (int i = 0; i < BITMAP_SIZE; i++) {
                sumPerClass[instance][i] += row.get(i);
            }
        }

        // Calculate centroid for each class
        for (int digit = 0; digit < 10; digit++) {
            if (countPerClass[digit] > 0) {
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    centroids[digit][i] = sumPerClass[digit][i] / countPerClass[digit];
                }
            }
        }
        return centroids;
    }

    // Distance from Centroid Algorythm
    private static class DistanceFromCentroid implements Algorithm {
        
        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            double[][] centroids = calculateCentroids(trainingSet);

            double minDistance = Double.MAX_VALUE;
            int closestClass = -1;

            for (int digit = 0; digit < 10; digit++) {
                double sum = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double diff = sample.get(i) - centroids[digit][i];
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

    // Support Vector Machine Algorythm
    private static class SupportVectorMachine implements Algorithm {
        private static final int CLASSES = 10;
        private static final int EXTRA_FEATURES = CLASSES;
        private static final int FEATURE_SIZE = BITMAP_SIZE + EXTRA_FEATURES;
        private static final int MAX_EPOCHS = 10; // after 10 it dips a little and then flatlines
        private static final double LEARNING_RATE = 0.02;
        private static final double MARGIN = 0.002; // minimum score gap before we accept a prediction 10 times less than learning rate seems to work best
        private static final long RANDOM_SEED = 42;


        private double[][] centroidCache;
        private double[] featureMeans = new double[FEATURE_SIZE]; // per-feature mean for normalization
        private double[] featureStdDevs = new double[FEATURE_SIZE]; // per-feature standard deviation for normalization (the square root of variance describing how far feature values spread from their mean.)
        private double[][] oneVsRestWeights = new double[CLASSES][FEATURE_SIZE];
        private double[] oneVsRestBias = new double[CLASSES];
        private List<LinearPerceptron> pairwiseClassifiers = new ArrayList<>();
        private boolean trained = false;

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (!trained) {
                train(trainingSet);
                trained = true;
            }
            double[] features = normalizeFeatureVector(projectToFeatureSpace(sample, centroidCache));
            int oneVsRestPrediction = classifyOneVsRest(features, null);
            int oneVsOnePrediction = classifyOneVsOne(features);
            return new int[] {oneVsRestPrediction, oneVsOnePrediction};
        }

        private void train(List<List<Integer>> trainingSet) {
            centroidCache = calculateCentroids(trainingSet);

            double[][] featureMatrix = new double[trainingSet.size()][FEATURE_SIZE];
            int[] classes_array = new int[trainingSet.size()];
            for (int i = 0; i < trainingSet.size(); i++) {
                List<Integer> row = trainingSet.get(i);
                featureMatrix[i] = projectToFeatureSpace(row, centroidCache);
                classes_array[i] = row.get(BITMAP_SIZE);
            }

            computeNormalizationStats(featureMatrix); // collect mean/standard deviation for every feature
            applyNormalization(featureMatrix); // normalize the entire training dataset

            trainOneVsRest(featureMatrix, classes_array);
            trainOneVsOne(featureMatrix, classes_array);
        }

        private void trainOneVsRest(double[][] features, int[] classes_array) {
            oneVsRestWeights = new double[CLASSES][FEATURE_SIZE];
            oneVsRestBias = new double[CLASSES];

            double[][] weightSums = new double[CLASSES][FEATURE_SIZE];
            double[] biasSums = new double[CLASSES];
            long steps = 0;

            // array of indexes to shuffle
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < features.length; i++) {
                indices.add(i);
            }
            Random random = new Random(RANDOM_SEED);

            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                Collections.shuffle(indices, random);

                int mistakes = 0;
                // training loop: picks shafled indexes and trains weights on them
                for (int idx : indices) {
                    double[] sample = features[idx];
                    int target = classes_array[idx];

                    double[] scores = new double[CLASSES];
                    int predicted = classifyOneVsRest(sample, scores);

                    double targetScore = scores[target];
                    double rivalScore = Double.NEGATIVE_INFINITY;
                    int rivalClass = -1;
                    for (int c = 0; c < CLASSES; c++) {
                        if (c == target) {
                            continue;
                        }
                        if (scores[c] > rivalScore) {
                            rivalScore = scores[c];
                            rivalClass = c;
                        }
                    }

                    boolean violation = predicted != target || (targetScore - rivalScore) <= MARGIN;
                    if (violation && rivalClass >= 0) {
                        updateOneVsRestWeights(sample, target, rivalClass);
                        mistakes++;
                    }

                    steps++;
                    for (int c = 0; c < CLASSES; c++) {
                        for (int f = 0; f < FEATURE_SIZE; f++) {
                            weightSums[c][f] += oneVsRestWeights[c][f];
                        }
                        biasSums[c] += oneVsRestBias[c];
                    }
                }

                if (mistakes == 0) {
                    break;
                }
            }

            // average weights over all steps
            steps = Math.max(steps, 1);
            for (int c = 0; c < CLASSES; c++) {
                for (int f = 0; f < FEATURE_SIZE; f++) {
                    oneVsRestWeights[c][f] = weightSums[c][f] / steps;
                }
                oneVsRestBias[c] = biasSums[c] / steps;
            }
        }

        private void trainOneVsOne(double[][] features, int[] classes_array) {
            pairwiseClassifiers = new ArrayList<>();

            for (int classA = 0; classA < CLASSES; classA++) {
                for (int classB = classA + 1; classB < CLASSES; classB++) {
                    List<Integer> classAIndices = new ArrayList<>(); // ammount of samples for class A
                    List<Integer> classBIndices = new ArrayList<>(); // ammount of samples for class B

                    // count samples for both classes
                    for (int i = 0; i < features.length; i++) {
                        if (classes_array[i] == classA) {
                            classAIndices.add(i);
                        } else if (classes_array[i] == classB) {
                            classBIndices.add(i);
                        }
                    }

                    if (classAIndices.isEmpty() || classBIndices.isEmpty()) {
                        continue;
                    }

                    LinearPerceptron perceptron = new LinearPerceptron(classA, classB, FEATURE_SIZE);
                    Random random = new Random(RANDOM_SEED + classA * CLASSES + classB);

                    // training loop for one-vs-one perceptron
                    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                        // shufeling eliminates bias (funnly in testing these 2 lines of shufeling made about 1% positive difference in success rate)
                        Collections.shuffle(classAIndices, random);
                        Collections.shuffle(classBIndices, random);

                        boolean updated = false;
                        int maxSize = Math.max(classAIndices.size(), classBIndices.size());

                        for (int offset = 0; offset < maxSize; offset++) {
                            if (offset < classAIndices.size()) {
                                updated |= perceptron.perceptronLearning(features[classAIndices.get(offset)], 1); // |= is logical OR assignment operator
                            }
                            if (offset < classBIndices.size()) {
                                updated |= perceptron.perceptronLearning(features[classBIndices.get(offset)], -1);
                            }
                        }

                        if (!updated) {
                            break;
                        }
                    }

                    // replace raw weights with their averaged counterpart
                    perceptron.finalizeAverage(); 
                    pairwiseClassifiers.add(perceptron);
                }
            }
        }

        // method projecting original bitmap into feature space adding distances from centroids
        private double[] projectToFeatureSpace(List<Integer> sample, double[][] centroids) {
            double[] features = new double[FEATURE_SIZE];

            for (int i = 0; i < BITMAP_SIZE; i++) {
                features[i] = sample.get(i);
            }

            for (int digit = 0; digit < CLASSES; digit++) {
                double sum = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double diff = sample.get(i) - centroids[digit][i];
                    sum += diff * diff;
                }
                features[BITMAP_SIZE + digit] = Math.sqrt(sum);
            }

            return features;
        }

        // method updating weights for one-vs-rest classifiers
        private void updateOneVsRestWeights(double[] features, int targetClass, int rivalClass) {
            for (int i = 0; i < FEATURE_SIZE; i++) {
                oneVsRestWeights[targetClass][i] += LEARNING_RATE * features[i];
                oneVsRestWeights[rivalClass][i] -= LEARNING_RATE * features[i];
            }
            oneVsRestBias[targetClass] += LEARNING_RATE;
            oneVsRestBias[rivalClass] -= LEARNING_RATE;
        }

        // method classifying sample using one-vs-rest strategy
        private int classifyOneVsRest(double[] features, double[] scoreBuffer) {
            double maxScore = Double.NEGATIVE_INFINITY;
            int bestClass = 0;

            for (int c = 0; c < CLASSES; c++) {
                double score = oneVsRestBias[c];
                for (int i = 0; i < FEATURE_SIZE; i++) {
                    score += oneVsRestWeights[c][i] * features[i];
                }
                if (scoreBuffer != null) {
                    scoreBuffer[c] = score;
                }
                if (score > maxScore) {
                    maxScore = score;
                    bestClass = c;
                }
            }

            return bestClass;
        }

        // method classifying sample using one-vs-one strategy
        private int classifyOneVsOne(double[] features) {
            int[] votes = new int[CLASSES];
            for (LinearPerceptron perceptron : pairwiseClassifiers) {
                votes[perceptron.predict(features)]++;
            }
            int bestClass = 0;
            int bestVotes = -1;
            for (int c = 0; c < CLASSES; c++) {
                if (votes[c] > bestVotes) {
                    bestVotes = votes[c];
                    bestClass = c;
                }
            }
            return bestClass;
        }

        // normalization methods squashing data into 0-1 range for each feature minimasing larger distances impacting results \/
        // method computeing mean and standard deviation for each feature across the dataset
        private void computeNormalizationStats(double[][] features) {
            for (int f = 0; f < FEATURE_SIZE; f++) {
                double sum = 0;
                double sumSq = 0;
                for (double[] vector : features) {
                    sum += vector[f];
                    sumSq += vector[f] * vector[f];
                }
                double mean = sum / features.length;
                double variance = (sumSq / features.length) - (mean * mean);
                featureMeans[f] = mean;
                featureStdDevs[f] = Math.max(Math.sqrt(Math.max(variance, 0)), 1e-9);
            }
        }

        // method applying normalization to the training dataset
        private void applyNormalization(double[][] features) {
            for (double[] vector : features) {
                for (int f = 0; f < FEATURE_SIZE; f++) {
                    vector[f] = (vector[f] - featureMeans[f]) / featureStdDevs[f];
                }
            }
        }

        // method normalizing a single bitmap (used for samle from test set during prediction)
        private double[] normalizeFeatureVector(double[] features) {
            double[] normalized = new double[FEATURE_SIZE];
            for (int f = 0; f < FEATURE_SIZE; f++) {
                normalized[f] = (features[f] - featureMeans[f]) / featureStdDevs[f];
            }
            return normalized;
        }

        // inner class representing a linear perceptron for one-vs-one classification
        private static class LinearPerceptron {
            final int positiveClass;
            final int negativeClass;
            final double[] weights;
            double bias;

            private final double[] weightSum;
            private double biasSum;
            private long steps;

            LinearPerceptron(int positiveClass, int negativeClass, int featureSize) {
                this.positiveClass = positiveClass;
                this.negativeClass = negativeClass;
                this.weights = new double[featureSize];
                this.weightSum = new double[featureSize];
            }

            // perceptron learning rule: if misclassified, update weights and bias
            boolean perceptronLearning(double[] features, int target) {
                double activation = bias;
                for (int i = 0; i < weights.length; i++) {
                    activation += weights[i] * features[i];
                }

                boolean violation = target * activation <= MARGIN; // enforce margin during binary training
                if (violation) {
                    for (int i = 0; i < weights.length; i++) {
                        weights[i] += LEARNING_RATE * target * features[i];
                    }
                    bias += LEARNING_RATE * target;
                }

                steps++;
                for (int i = 0; i < weights.length; i++) {
                    weightSum[i] += weights[i];
                }
                biasSum += bias;

                return violation;
            }

            void finalizeAverage() {
                if (steps == 0) {
                    return;
                }
                for (int i = 0; i < weights.length; i++) {
                    weights[i] = weightSum[i] / steps; // replace with averaged weights
                }
                bias = biasSum / steps;
            }

            int predict(double[] features) {
                double activation = bias;
                for (int i = 0; i < weights.length; i++) {
                    activation += weights[i] * features[i];
                }
                return activation >= 0 ? positiveClass : negativeClass;
            }
        }
    }

    // K-Nearest Neighbour Algorythm
    private static class K_NEAREST_NEIGHBOUR implements Algorithm {

        private static final int K = 3;

         @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            PriorityQueue<double[]> heap = new PriorityQueue<>((a, b) -> Double.compare(b[0], a[0]));
            // calculate euclidean distance from the sample to all training set candidates
            for (List<Integer> candidate : trainingSet) {
                double distance = 0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double diff = sample.get(i) - candidate.get(i);
                    distance += diff * diff; // squared distance gives better results than square root of distance
                }
                // fill the heap with closest K neighbours
                if (heap.size() < K) {
                    heap.offer(new double[] {distance, candidate.get(BITMAP_SIZE)});
                } else if (distance < heap.peek()[0]) {
                    heap.poll();
                    heap.offer(new double[] {distance, candidate.get(BITMAP_SIZE)});
                }
            }
            // find the most common class among the K neighbours
            int[] votes = new int[10];
            while (!heap.isEmpty()) {
                int digit = (int) heap.poll()[1];
                votes[digit]++;
            }
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

    // K-Means Algorythm
    private static class KMeans implements Algorithm {

        private static final int CLUSTERS = 10; // Number of clusters (digits 0-9)
        private static final int MAX_ITERATIONS = 1; // Maximum number of iterations for convergence

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            // Initialize centroids randomly from the training set
            double[][] centroids = initializeCentroids(trainingSet);

            // Perform the K-Means clustering algorithm
            for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
                // Assign each sample to the nearest centroid
                int[] assignments = assignSamplesToCentroids(trainingSet, centroids);

                // Recalculate centroids based on the assignments
                double[][] newCentroids = recalculateCentroids(trainingSet, assignments);

                // Check for convergence (if centroids do not change)
                if (hasConverged(centroids, newCentroids)) {
                    break;
                }

                centroids = newCentroids;
            }

            // Predict the cluster for the given sample
            return Integer.valueOf(predictCluster(sample, centroids));
        }

        // Initialize centroids randomly from the training set
        private double[][] initializeCentroids(List<List<Integer>> trainingSet) {
            Random random = new Random();
            double[][] centroids = new double[CLUSTERS][BITMAP_SIZE];
            for (int i = 0; i < CLUSTERS; i++) {
                List<Integer> randomSample = trainingSet.get(random.nextInt(trainingSet.size()));
                for (int j = 0; j < BITMAP_SIZE; j++) {
                    centroids[i][j] = randomSample.get(j);
                }
            }
            return centroids;
        }

        // Assign each sample to the nearest centroid
        private int[] assignSamplesToCentroids(List<List<Integer>> trainingSet, double[][] centroids) {
            int[] assignments = new int[trainingSet.size()];
            for (int i = 0; i < trainingSet.size(); i++) {
                List<Integer> sample = trainingSet.get(i);
                assignments[i] = predictCluster(sample, centroids);
            }
            return assignments;
        }

        // Recalculate centroids based on the current assignments
        private double[][] recalculateCentroids(List<List<Integer>> trainingSet, int[] assignments) {
            double[][] centroids = new double[CLUSTERS][BITMAP_SIZE];
            int[] counts = new int[CLUSTERS];

            for (int i = 0; i < trainingSet.size(); i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                List<Integer> sample = trainingSet.get(i);
                for (int j = 0; j < BITMAP_SIZE; j++) {
                    centroids[cluster][j] += sample.get(j);
                }
            }

            for (int cluster = 0; cluster < CLUSTERS; cluster++) {
                if (counts[cluster] > 0) {
                    for (int j = 0; j < BITMAP_SIZE; j++) {
                        centroids[cluster][j] /= counts[cluster];
                    }
                }
            }

            return centroids;
        }

        // Check if centroids have converged
        private boolean hasConverged(double[][] oldCentroids, double[][] newCentroids) {
            for (int i = 0; i < CLUSTERS; i++) {
                for (int j = 0; j < BITMAP_SIZE; j++) {
                    if (Math.abs(oldCentroids[i][j] - newCentroids[i][j]) > 1e-6) {
                        return false;
                    }
                }
            }
            return true;
        }

        // Predict the cluster for a given sample
        private int predictCluster(List<Integer> sample, double[][] centroids) {
            double minDistance = Double.MAX_VALUE;
            int bestCluster = -1;

            for (int cluster = 0; cluster < CLUSTERS; cluster++) {
                double distance = 0;
                for (int j = 0; j < BITMAP_SIZE; j++) {
                    double diff = sample.get(j) - centroids[cluster][j];
                    distance += diff * diff;
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = cluster;
                }
            }

            return bestCluster;
        }
    }

    // Mahalanobis Distance Algorithm
    private static class MahalanobisDistance implements Algorithm {
        private static final int CLASSES = 10;

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            // Get the number of features in the sample
            int featureCount = getFeatureCount(sample);
            if (featureCount <= 0) {
                return Integer.valueOf(-1); // Return -1 if the feature count is invalid
            }

            // Calculate centroids for each class
            double[][] centroids = calculateCentroids(trainingSet);

            // Count the number of samples per class
            int[] classCounts = new int[CLASSES];
            for (List<Integer> row : trainingSet) {
                int label = getLabel(row);
                if (label >= 0 && label < CLASSES) {
                    classCounts[label]++;
                }
            }

            // Compute the feature means and covariance matrix
            double[] featureMeans = computeFeatureMeans(trainingSet, featureCount);
            double[][] covariance = computeCovarianceMatrix(trainingSet, featureCount, featureMeans);
            double[][] inverseCovariance = invertMatrix(covariance);
            if (inverseCovariance == null) {
                return Integer.valueOf(-1); // Return -1 if the covariance matrix is not invertible
            }

            // Convert the sample into a vector
            double[] sampleVector = new double[featureCount];
            for (int i = 0; i < featureCount; i++) {
                sampleVector[i] = sample.get(i);
            }

            // Calculate the Mahalanobis distance to each class centroid
            double bestDistance = Double.MAX_VALUE;
            int bestClass = -1;
            for (int digit = 0; digit < CLASSES; digit++) {
                if (classCounts[digit] == 0 || centroids[digit].length == 0) {
                    continue; // Skip classes with no samples
                }
                double[] diff = new double[featureCount];
                for (int i = 0; i < featureCount && i < centroids[digit].length; i++) {
                    diff[i] = sampleVector[i] - centroids[digit][i];
                }
                double distance = computeMahalanobisDistance(diff, inverseCovariance);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClass = digit;
                }
            }

            // Return the predicted class or 0 if no valid class was found
            return Integer.valueOf(bestClass >= 0 ? bestClass : 0);
        }

        // Helper method to get the number of features in a sample
        private int getFeatureCount(List<Integer> sample) {
            return sample != null ? sample.size() : 0;
        }

        // Helper method to get the label (class) of a sample
        private int getLabel(List<Integer> row) {
            return row != null && row.size() > BITMAP_SIZE ? row.get(BITMAP_SIZE) : -1;
        }
    // } could end here but I moved it down for code organisation the helper functions are used in this class only anyway

        // Method to compute the mean of each feature across the training set
        private static double[] computeFeatureMeans(List<List<Integer>> trainingSet, int featureCount) {
            double[] means = new double[featureCount];
            if (trainingSet == null || trainingSet.isEmpty()) {
                return means;
            }
            for (List<Integer> row : trainingSet) {
                for (int i = 0; i < featureCount; i++) {
                    means[i] += row.get(i);
                }
            }
            int n = trainingSet.size();
            for (int i = 0; i < featureCount; i++) {
                means[i] /= n;
            }
            return means;
        }

        // Method to compute the covariance matrix of the features in the training set
        private static double[][] computeCovarianceMatrix(List<List<Integer>> trainingSet, int featureCount, double[] means) {
            double[][] covariance = new double[featureCount][featureCount];
            if (trainingSet == null || trainingSet.size() <= 1) {
                for (int i = 0; i < featureCount; i++) {
                    covariance[i][i] = 1.0;
                }
                return covariance;
            }
            for (List<Integer> row : trainingSet) {
                for (int i = 0; i < featureCount; i++) {
                    double diffI = row.get(i) - means[i];
                    for (int j = 0; j < featureCount; j++) {
                        double diffJ = row.get(j) - means[j];
                        covariance[i][j] += diffI * diffJ;
                    }
                }
            }
            double denom = trainingSet.size() - 1.0;
            for (int i = 0; i < featureCount; i++) {
                for (int j = 0; j < featureCount; j++) {
                    covariance[i][j] /= denom;
                }
                covariance[i][i] += 1e-6; // regularization
            }
            return covariance;
        }

        // Method to invert a matrix using Gaussian elimination
        private static double[][] invertMatrix(double[][] matrix) {
            int n = matrix.length;
            double[][] augmented = new double[n][2 * n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(matrix[i], 0, augmented[i], 0, n);
                augmented[i][i + n] = 1.0;
            }

            for (int col = 0; col < n; col++) {
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
                    return null;
                }
                if (pivot != col) {
                    double[] tmp = augmented[pivot];
                    augmented[pivot] = augmented[col];
                    augmented[col] = tmp;
                }

                double pivotVal = augmented[col][col];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[col][j] /= pivotVal;
                }

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

            double[][] inverse = new double[n][n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(augmented[i], n, inverse[i], 0, n);
            }
            return inverse;
        }

        // Method to compute the Mahalanobis distance given the difference vector and inverse covariance matrix
        private static double computeMahalanobisDistance(double[] diff, double[][] inverseCovariance) {
            double[] intermediate = new double[diff.length];
            for (int i = 0; i < diff.length; i++) {
                double sum = 0;
                for (int j = 0; j < diff.length; j++) {
                    sum += inverseCovariance[i][j] * diff[j];
                }
                intermediate[i] = sum;
            }
            double distance = 0;
            for (int i = 0; i < diff.length; i++) {
                distance += diff[i] * intermediate[i];
            }
            return Math.sqrt(Math.max(distance, 0));
        }
    }

    // Genetic Algorythm
    private static class GeneticAlgorythm implements Algorithm {

        private static final int POPULATION_SIZE = 50; // Number of individuals in the population
        private static final int GENERATIONS = 10; // Number of generations to evolve
        private static final double MUTATION_RATE = 0.1; // Probability of mutation
        private static final int CLASSES = 10; // Number of classes (digits 0-9)

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            // Initialize the population randomly
            List<double[]> population = initializePopulation();

            // Evolve the population over generations
            for (int generation = 0; generation < GENERATIONS; generation++) {
                // Evaluate the fitness of each individual
                double[] fitness = evaluateFitness(population, trainingSet);

                // Select parents and create the next generation
                List<double[]> nextGeneration = new ArrayList<>();
                for (int i = 0; i < POPULATION_SIZE; i++) {
                    double[] parent1 = selectParent(population, fitness);
                    double[] parent2 = selectParent(population, fitness);
                    double[] child = crossover(parent1, parent2);
                    mutate(child);
                    nextGeneration.add(child);
                }

                population = nextGeneration;
            }

            // Predict the class for the given sample using the best individual
            double[] bestIndividual = population.get(0);
            return Integer.valueOf(predictClass(sample, bestIndividual));
        }

        // Initialize the population randomly
        private List<double[]> initializePopulation() {
            Random random = new Random();
            List<double[]> population = new ArrayList<>();
            for (int i = 0; i < POPULATION_SIZE; i++) {
                double[] individual = new double[BITMAP_SIZE];
                for (int j = 0; j < BITMAP_SIZE; j++) {
                    individual[j] = random.nextDouble();
                }
                population.add(individual);
            }
            return population;
        }

        // Evaluate the fitness of each individual
        private double[] evaluateFitness(List<double[]> population, List<List<Integer>> trainingSet) {
            double[] fitness = new double[population.size()];
            for (int i = 0; i < population.size(); i++) {
                double[] individual = population.get(i);
                int correct = 0;
                for (List<Integer> sample : trainingSet) {
                    int actualClass = sample.get(BITMAP_SIZE);
                    int predictedClass = predictClass(sample, individual);
                    if (actualClass == predictedClass) {
                        correct++;
                    }
                }
                fitness[i] = correct / (double) trainingSet.size();
            }
            return fitness;
        }

        // Select a parent using roulette wheel selection
        private double[] selectParent(List<double[]> population, double[] fitness) {
            double totalFitness = 0;
            for (double f : fitness) {
                totalFitness += f;
            }

            double randomValue = Math.random() * totalFitness;
            double cumulativeFitness = 0;
            for (int i = 0; i < population.size(); i++) {
                cumulativeFitness += fitness[i];
                if (cumulativeFitness >= randomValue) {
                    return population.get(i);
                }
            }

            return population.get(population.size() - 1);
        }

        // Perform crossover between two parents
        private double[] crossover(double[] parent1, double[] parent2) {
            double[] child = new double[BITMAP_SIZE];
            for (int i = 0; i < BITMAP_SIZE; i++) {
                child[i] = Math.random() < 0.5 ? parent1[i] : parent2[i];
            }
            return child;
        }

        // Mutate an individual
        private void mutate(double[] individual) {
            Random random = new Random();
            for (int i = 0; i < BITMAP_SIZE; i++) {
                if (random.nextDouble() < MUTATION_RATE) {
                    individual[i] += random.nextGaussian() * 0.1; // Add small random noise
                }
            }
        }

        // Predict the class for a given sample using an individual
        private int predictClass(List<Integer> sample, double[] individual) {
            double score = 0;
            for (int i = 0; i < BITMAP_SIZE; i++) {
                score += sample.get(i) * individual[i];
            }
            return (int) Math.round(score) % CLASSES;
        }
    }

    // All Algorythms at above at once  
    private static class AllAtOnce implements Algorithm {

        @Override
        public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            // List of all algorithms to run
            Algorithm[] algorithms = {
                EUCLIDEAN_DISTANCE,
                MULTI_LAYER_PERCEPTRON,
                DISTANCE_FROM_CENTROID,
                SUPPORT_VECTOR_MACHINE,
                K_NEAREST_NEIGHBOUR,
                MAHALANOBIS_DISTANCE
            };

            // Array to store votes for each class (0-9)
            int[] votes = new int[10];

            // Run each algorithm and collect its predictions
            for (Algorithm algorithm : algorithms) {
                    Object result = algorithm.predict(sample, trainingSet);

                    // Handle algorithms that return int[] (e.g., SupportVectorMachine)
                    if (result instanceof int[]) {
                        int[] predictions = (int[]) result;
                        for (int predictedClass : predictions) {
                            if (predictedClass >= 0 && predictedClass < votes.length) {
                                // to see what each algorythm voted for uncomment bellow. 
                                // (expected behavour would be waiting long time for mlp and svm to train and then it floods the console with all (2810*num of algorythms used) votes :D)
                                // in my tests the algorythms sometimes confused 2 and 6, 3 and 8, and 3 and 5
                                // offcourse it might be sample dependent but in most cases there were multiple algorythms voteing for example for 2  and multiple voteing for 6 about the same sample
                                // System.out.println("Algorythm " + algorithm.getClass().getSimpleName() + " voted: " + predictedClass);
                                votes[predictedClass]++;
                            }
                        }
                    } 
                    // Handle algorithms that return a single Integer
                    else if (result instanceof Integer) {
                        int predictedClass = (Integer) result;
                        if (predictedClass >= 0 && predictedClass < votes.length) {
                            // System.out.println("Algorythm " + algorithm.getClass().getSimpleName() + "voted: " + predictedClass);
                            votes[predictedClass]++;
                        }
                    } else {
                        System.err.println("Unexpected result type from algorithm: " + algorithm.getClass().getSimpleName());
                    }
            }

            // Find the class with the highest votes
            int bestClass = 0;
            int maxVotes = 0;
            for (int i = 0; i < votes.length; i++) {
                if (votes[i] > maxVotes) {
                    maxVotes = votes[i];
                    bestClass = i;
                }
            }

            // Return the class with the most votes
            return Integer.valueOf(bestClass);
        }
    }

    // function to evaluate success rate of inputed algorithm
    private static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        Thread animationThread = new Thread(() -> showLoadingAnimation());
        animationThread.start(); // Start the animation in a separate thread

        // Start timing
        long startTime = System.nanoTime();

        if (algorithm instanceof SupportVectorMachine svm) {
            int correctOneVsRest = 0;
            int correctOneVsOne = 0;
            for (List<Integer> sample : dataSetB) {
                int actualDigit = sample.get(BITMAP_SIZE);
                int[] prediction = (int[]) svm.predict(sample, dataSetA);
                if (prediction[0] == actualDigit) correctOneVsRest++;
                if (prediction[1] == actualDigit) correctOneVsOne++;
            }
            double size = dataSetB.size();
            System.out.println("\n--- " + label + " Success Rate ---");
            System.out.println("1-vs-Rest Correct: " + correctOneVsRest + " / " + dataSetB.size());
            System.out.println("1-vs-Rest Success Rate: " + (correctOneVsRest / size) * 100 + "%");
            System.out.println("1-vs-1 Correct: " + correctOneVsOne + " / " + dataSetB.size());
            System.out.println("1-vs-1 Success Rate: " + (correctOneVsOne / size) * 100 + "%");

            // End timing and print elapsed time
            long endTime = System.nanoTime();
            System.out.println("Evaluation Time: " + (endTime - startTime) / 1_000_000 + " ms");
            return;
        }

        if (algorithm instanceof MultiLayerPerceptron) {
            System.out.println("\n--- " + label + " parameters used in calculation ---");
            System.out.println(((MultiLayerPerceptron) algorithm).getParameters());
        }

        int correctMatches = 0;
        for (int s = 0; s < dataSetB.size(); s++) {
            List<Integer> sample = dataSetB.get(s);
            int actualDigit = sample.get(BITMAP_SIZE);
            int predictedDigit = (Integer) algorithm.predict(sample, dataSetA);
            if (actualDigit == predictedDigit) {
                correctMatches++;
            }
        }

        double successRate = (correctMatches / (double) dataSetB.size()) * 100;
        System.out.println("\n--- " + label + " Success Rate ---");
        System.out.println("Correct Matches: " + correctMatches + " / " + dataSetB.size());
        System.out.println("Success Rate: " + successRate + "%");

        // Stop the animation
        animationThread.interrupt();
        // End timing and print elapsed time
        long endTime = System.nanoTime();
        System.out.println("Evaluation Time: " + (endTime - startTime) / 1_000_000_000.0 + " seconds");
    }

    // Function to display a simple animation with dots
    private static void showLoadingAnimation() {
        try {
            String[] frames = {".  ", ".. ", "..."};
            while (true) {
                for (String frame : frames) {
                    System.out.print("\revaluating" + frame);
                    Thread.sleep(500);
                }
                System.out.print("\r             ");
                Thread.sleep(500);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // function to read the csv files 
    private static List<List<Integer>> readCsvFile(String dataSetFilePath) {
        List<List<Integer>> dataSet = new ArrayList<>();
        int rowCount = 0;

        System.out.println("Starting to read CSV file: " + dataSetFilePath);
        try (Scanner scanner = new Scanner(new java.io.File(dataSetFilePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] values = line.split(",");

                // Ensure the row has the expected number of columns
                if (values.length != ENTIRE_BITMAP_SIZE) {
                    System.err.println("Warning: Row " + (rowCount + 1) + " has " + values.length + " columns, expected " + ENTIRE_BITMAP_SIZE + ". Skipping row.");
                    continue;
                }

                List<Integer> currentRow = new ArrayList<>();
                boolean conversionError = false;

                for (int i = 0; i < ENTIRE_BITMAP_SIZE; i++) {
                    try {
                        currentRow.add(Integer.parseInt(values[i].trim()));
                    } catch (NumberFormatException error) {
                        System.err.println("Error: Could not convert value '" + values[i] + "' to integer in row " + (rowCount + 1) + ". Skipping row.");
                        conversionError = true;
                        break;
                    }
                }

                if (!conversionError) {
                    dataSet.add(currentRow);
                    rowCount++;
                }
            }
            return dataSet;

        } catch (IOException error) {
            System.err.println("\n--- ERROR: Failed to read the file ---");
            System.err.println("Ensure the file exists at the correct path: " + dataSetFilePath);
            System.err.println("Details: " + error.getMessage());
            return null;
        }
    }

    // function that prints entire dataset
    private static void printDataSet(List<List<Integer>> dataSet) {
        for (int i = 0; i < dataSet.size(); i++) {
            List<Integer> row = dataSet.get(i);
            
            String prefix = "Bitmap representation of a digit " + (i + 1) + ": [";
            System.out.print(prefix);
            
            for (int j = 0; j < row.size(); j++) {
                System.out.print(row.get(j));
                
                if (j < row.size() - 1) {
                    System.out.print(", ");
                }
                
                if ((j + 1) % 10 == 0 && j < row.size() - 1) {
                    System.out.print("\n" + " ".repeat(prefix.length()));
                }
            }
            System.out.println("]");
        }
    }
    
    // function to read selected ammount of bitmaps 
    private static void printLimitedDataSet(List<List<Integer>> dataSet) {
        for (int i = 0; i < BITMAPS_TO_DISPLAY; i++) {
            List<Integer> row = dataSet.get(i);
            
            String prefix = "Bitmap representation of a digit " + (i + 1) + ": [";
            System.out.print(prefix);
            
            for (int j = 0; j < row.size(); j++) {
                System.out.print(row.get(j));
                
                if (j < row.size() - 1) {
                    System.out.print(", ");
                }
                
                if ((j + 1) % 10 == 0 && j < row.size() - 1) {
                    System.out.print("\n" + " ".repeat(prefix.length()));
                }
            }
            System.out.println("]");
        }
    }
        
    // user subinterface for prints
    private static void PrintDataUserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB, Scanner scanner) {
        boolean running = true;
        
        while (running) {
        System.out.println("\n=== Print Actions: ===");
        System.out.println("1 -> Print entire data set A");
        System.out.println("2 -> Print entire data set B");
        System.out.println("3 -> Print " + BITMAPS_TO_DISPLAY + " bitmaps from data set A");
        System.out.println("4 -> Print " + BITMAPS_TO_DISPLAY + " bitmaps from data set B");
        System.out.println("0 -> Exit");
        System.out.print("\nEnter your choice (0-4): ");
        try {
            int choice = scanner.nextInt();
            
            switch (choice) {case 1:
                if (dataSetA != null && !dataSetA.isEmpty()) {
                    System.out.println("\n--- DataSet A (Complete) ---");
                    System.out.println("Total rows: " + dataSetA.size());
                    printDataSet(dataSetA);
                } else {
                    System.out.println("\nNo data (DataSetA) was loaded or an error occurred during reading.");
                }
                break;
                
            case 2:
                if (dataSetB != null && !dataSetB.isEmpty()) {
                    System.out.println("\n--- DataSet B (Complete) ---");
                    System.out.println("Total rows: " + dataSetB.size());
                    printDataSet(dataSetB);
                } else {
                    System.out.println("\nNo data (DataSetB) was loaded or an error occurred during reading.");
                }
                break;
                
            case 3:
                if (dataSetA != null && !dataSetA.isEmpty()) {
                    System.out.println("\n--- DataSet A (First " + BITMAPS_TO_DISPLAY + " bitmaps) ---");
                    System.out.println("Total rows: " + dataSetA.size());
                    printLimitedDataSet(dataSetA);
                } else {
                    System.out.println("\nNo data (DataSetA) was loaded or an error occurred during reading.");
                }
                break;
                
            case 4:
                if (dataSetB != null && !dataSetB.isEmpty()) {
                    System.out.println("\n--- DataSet B (First " + BITMAPS_TO_DISPLAY + " bitmaps) ---");
                    System.out.println("Total rows: " + dataSetB.size());
                    printLimitedDataSet(dataSetB);
                } else {
                    System.out.println("\nNo data (DataSetB) was loaded or an error occurred during reading.");
                }
                break;
                
            case 0:
                System.out.println("\nExiting");
                running = false;
                break;
                
                default:
                    System.out.println("\nInvalid choice. Please enter a number corresponting to avaliable actions.");
            }
            
            } catch (Exception error) {
            System.out.println("\nInvalid input. Please enter a number corresponting to avaliable actions.");
            scanner.nextLine(); // Clear the invalid input
            }
        }
    }
    
    // user interface 
    private static void UserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB) {
    	Scanner scanner = new Scanner(System.in); 
        boolean running = true;
        
        while (running) {
            System.out.println("\n=== Actions: ===");
            System.out.println("1 -> Printing datasets options");
            System.out.println("2 -> Euclidean Distance");
            System.out.println("3 -> Multi Layer Perceptron");
            System.out.println("4 -> Distance From Centroid");
            System.out.println("5 -> Support Vector Machine");
            System.out.println("6 -> K Nearest Neighbour");
            System.out.println("7 -> K Means");
            System.out.println("8 -> Mahalanobis Distance");
            System.out.println("9 -> Genetic Algorythm");
            System.out.println("10 -> All at Once");
            System.out.println("0 -> Exit");
            System.out.print("\nEnter your choice (0-10): ");
            
            try {
                int choice = scanner.nextInt();
                
                switch (choice) {
                    case 1:
                        PrintDataUserInterface(dataSetA, dataSetB, scanner);
                        break;
                
                    case 2:
                        evaluateAlgorithm(dataSetA, dataSetB, EUCLIDEAN_DISTANCE, "Euclidean Distance"); // tarin on A, test on B
                        break;
                        
                    case 3:
                        evaluateAlgorithm(dataSetA, dataSetB, MULTI_LAYER_PERCEPTRON, "Multi Layer Perceptron"); // train on A, test on B
                        break;
                    case 4:
                    	evaluateAlgorithm(dataSetA, dataSetB, DISTANCE_FROM_CENTROID, "Distance From Centroid"); // train on A, test on B
                    	break;
                    
                    case 5:
                        evaluateAlgorithm(dataSetA, dataSetB, SUPPORT_VECTOR_MACHINE, "Support Vector Machine"); // train on A, test on B
                        break;

                    case 6:
                        evaluateAlgorithm(dataSetA, dataSetB, K_NEAREST_NEIGHBOUR, "K Nearest Neighbour"); // train on A, test on B
                        break;
                    
                    case 7:
                        evaluateAlgorithm(dataSetA, dataSetB, K_MEANS, "K_MEANS"); // train on A, test on B
                        break;

                    case 8:
                        evaluateAlgorithm(dataSetA, dataSetB, MAHALANOBIS_DISTANCE, "Mahalanobis Distance"); // train on A, test on B
                        break;

                    case 9:
                        evaluateAlgorithm(dataSetA, dataSetB, GENETIC_ALGORYTHM, "Genetic Algorythm"); // train on A, test on B
                        break;

                    case 10:
                        evaluateAlgorithm(dataSetA, dataSetB, ALL_AT_ONCE, "All at Once"); // train on A, test on B
                        break;

                    case 0:
                        System.out.println("\nExiting");
                        running = false;
                        break;
                        
                    default:
                        System.out.println("\nInvalid choice. Please enter a number corresponting to avaliable actions.");
                }
                
            } catch (Exception error) {
                System.out.println("\nInvalid input. Please enter a number corresponting to avaliable actions.");
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