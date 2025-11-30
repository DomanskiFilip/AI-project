package main;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
        int predict(List<Integer> sample, List<List<Integer>> trainingSet); // sample -> row to predict, trainingSet -> dataset A (usually)
    }

    private static final Algorithm EUCLIDEAN_DISTANCE = new EuclideanDistance();
    private static final Algorithm MULTI_LAYER_PERCEPTRON = new MultiLayerPerceptron();
    private static final Algorithm DISTANCE_FROM_CENTROID = new DistanceFromCentroid();
    private static final Algorithm SUPPORT_VECTOR_MACHINE = new SupportVectorMachine();

    // Euclidean Distance Algorythm
    private static class EuclideanDistance implements Algorithm {

        @Override
        public int predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (trainingSet == null || trainingSet.isEmpty()) {
                throw new IllegalArgumentException("Training dataset must not be null or empty.");
            }

            double minDistance = Double.MAX_VALUE;
            List<Integer> closest = null;

            for (int c = 0; c < trainingSet.size(); c++) {
                List<Integer> candidate = trainingSet.get(c);
                double sum = 0.0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double distance = sample.get(i) - candidate.get(i);
                    sum += distance * distance;
                }
                if (sum < minDistance) {
                    minDistance = sum;
                    closest = candidate;
                }
            }

            return closest != null ? closest.get(BITMAP_SIZE) : -1;
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
        public int predict(List<Integer> sample, List<List<Integer>> trainingSet) {
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
            return bestIndex;
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
                        double error = 0.0;
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

    private static class DistanceFromCentroid implements Algorithm {
        
        @Override
        public int predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            double[][] centroids = calculateCentroids(trainingSet);

            double minDistance = Double.MAX_VALUE;
            int closestClass = -1;

            for (int digit = 0; digit < 10; digit++) {
                double sum = 0.0;
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

            return closestClass;
        }

    }

    private static class SupportVectorMachine implements Algorithm {
        private static final int CLASSES = 10;
        private static final int EXTRA_FEATURES = CLASSES;
        private static final int FEATURE_SIZE = BITMAP_SIZE + EXTRA_FEATURES;
        private static final int MAX_EPOCHS = 500;
        private static final double LEARNING_RATE = 0.1;
        private static final long RANDOM_SEED = 42;

        private double[][] centroidCache;
        private double[][] oneVsRestWeights = new double[CLASSES][FEATURE_SIZE];
        private double[] oneVsRestBias = new double[CLASSES];
        private List<BinaryPerceptron> pairwiseClassifiers = new ArrayList<>();
        private boolean trained = false;

        @Override
        public int predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            return predictDetailed(sample, trainingSet).oneVsRest;
        }

        public PredictionResult predictDetailed(List<Integer> sample, List<List<Integer>> trainingSet) {
            if (!trained) {
                train(trainingSet);
                trained = true;
            }
            double[] features = projectToFeatureSpace(sample, centroidCache); // add 10 dimentions representing distance from centroids to sample
            int oneVsRestPrediction = classifyOneVsRest(features); // predict using 1-vs-rest
            int oneVsOnePrediction = pairwiseClassifiers.isEmpty() ? oneVsRestPrediction : classifyOneVsOne(features); // predict using 1-vs-1
            return new PredictionResult(oneVsRestPrediction, oneVsOnePrediction);
        }

        private void train(List<List<Integer>> trainingSet) {
            centroidCache = calculateCentroids(trainingSet);
            double[][] featureMatrix = new double[trainingSet.size()][FEATURE_SIZE];
            int[] labels = new int[trainingSet.size()];
            for (int i = 0; i < trainingSet.size(); i++) {
                List<Integer> row = trainingSet.get(i);
                featureMatrix[i] = projectToFeatureSpace(row, centroidCache);
                labels[i] = row.get(BITMAP_SIZE);
            }
            trainOneVsRest(featureMatrix, labels);
            trainOneVsOne(featureMatrix, labels);
        }

        // training with useing just one vs rest seperator
        private void trainOneVsRest(double[][] features, int[] labels) {
            oneVsRestWeights = new double[CLASSES][FEATURE_SIZE];
            oneVsRestBias = new double[CLASSES];
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < labels.length; i++) {
                indices.add(i);
            }
            Random rng = new Random(RANDOM_SEED);

            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                Collections.shuffle(indices, rng);
                int mistakes = 0;

                for (int idx : indices) {
                    double[] sample = features[idx];
                    int target = labels[idx];
                    int predicted = classifyOneVsRest(sample);
                    if (predicted != target) {
                        updateOneVsRestWeights(sample, target, predicted);
                        mistakes++;
                    }
                }

                if (mistakes == 0) {
                    break;
                }
            }
        }

        // training with useing one vs one seperators
        private void trainOneVsOne(double[][] features, int[] labels) {
            pairwiseClassifiers = new ArrayList<>();
            for (int classA = 0; classA < CLASSES; classA++) {
                for (int classB = classA + 1; classB < CLASSES; classB++) {
                    List<Integer> subset = new ArrayList<>();
                    boolean hasA = false;
                    boolean hasB = false;

                    for (int i = 0; i < labels.length; i++) {
                        if (labels[i] == classA) {
                            subset.add(i);
                            hasA = true;
                        } else if (labels[i] == classB) {
                            subset.add(i);
                            hasB = true;
                        }
                    }

                    if (!hasA || !hasB) {
                        continue;
                    }

                    BinaryPerceptron perceptron = new BinaryPerceptron(classA, classB, FEATURE_SIZE);
                    Random random = new Random(RANDOM_SEED + classA * CLASSES + classB);

                    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                        Collections.shuffle(subset, random);
                        boolean updated = false;

                        for (int index : subset) {
                            int target = labels[index] == classA ? 1 : -1;
                            double activation = perceptron.bias;
                            double[] sample = features[index];
                            for (int i = 0; i < FEATURE_SIZE; i++) {
                                activation += perceptron.weights[i] * sample[i];
                            }
                            if (target * activation <= 0) {
                                updated = true;
                                for (int i = 0; i < FEATURE_SIZE; i++) {
                                    perceptron.weights[i] += LEARNING_RATE * target * sample[i];
                                }
                                perceptron.bias += LEARNING_RATE * target;
                            }
                        }

                        if (!updated) {
                            break;
                        }
                    }

                    pairwiseClassifiers.add(perceptron);
                }
            }
        }

        // method to add 10 dimentions representing distance from centroids to sample
        private double[] projectToFeatureSpace(List<Integer> sample, double[][] centroids) {
            double[] features = new double[FEATURE_SIZE];
            for (int i = 0; i < BITMAP_SIZE; i++) {
                features[i] = sample.get(i);
            }
            for (int digit = 0; digit < CLASSES; digit++) {
                double sum = 0.0;
                for (int i = 0; i < BITMAP_SIZE; i++) {
                    double diff = sample.get(i) - centroids[digit][i];
                    sum += diff * diff;
                }
                features[BITMAP_SIZE + digit] = Math.sqrt(sum);
            }
            return features;
        }

        // classify useing trained linear separator for one vs rest
        private int classifyOneVsRest(double[] features) {
            double maxScore = Double.NEGATIVE_INFINITY;
            int bestClass = 0;
            for (int c = 0; c < CLASSES; c++) {
                double score = oneVsRestBias[c];
                for (int i = 0; i < FEATURE_SIZE; i++) {
                    score += oneVsRestWeights[c][i] * features[i];
                }
                if (score > maxScore) {
                    maxScore = score;
                    bestClass = c;
                }
            }
            return bestClass;
        }

        // classify useing trained linear separators for one vs one
        private int classifyOneVsOne(double[] features) {
            int[] votes = new int[CLASSES];
            for (BinaryPerceptron perceptron : pairwiseClassifiers) {
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

        private void updateOneVsRestWeights(double[] features, int targetClass, int predictedClass) {
            for (int i = 0; i < FEATURE_SIZE; i++) {
                oneVsRestWeights[targetClass][i] += LEARNING_RATE * features[i];
                oneVsRestWeights[predictedClass][i] -= LEARNING_RATE * features[i];
            }
            oneVsRestBias[targetClass] += LEARNING_RATE;
            oneVsRestBias[predictedClass] -= LEARNING_RATE;
        }

        private static class BinaryPerceptron {
            final int positiveClass;
            final int negativeClass;
            final double[] weights;
            double bias;

            BinaryPerceptron(int positiveClass, int negativeClass, int featureSize) {
                this.positiveClass = positiveClass;
                this.negativeClass = negativeClass;
                this.weights = new double[featureSize];
            }

            int predict(double[] features) {
                double activation = bias;
                for (int i = 0; i < weights.length; i++) {
                    activation += weights[i] * features[i];
                }
                return activation >= 0 ? positiveClass : negativeClass;
            }
        }

        private static class PredictionResult {
            final int oneVsRest;
            final int oneVsOne;

            PredictionResult(int oneVsRest, int oneVsOne) {
                this.oneVsRest = oneVsRest;
                this.oneVsOne = oneVsOne;
            }
        }
    }

    // function to evaluate success rate of inputed algorithm
    private static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        if (algorithm instanceof SupportVectorMachine svm) {
            int correctOneVsRest = 0;
            int correctOneVsOne = 0;
            for (List<Integer> sample : dataSetB) {
                int actualDigit = sample.get(BITMAP_SIZE);
                SupportVectorMachine.PredictionResult prediction = svm.predictDetailed(sample, dataSetA);
                if (prediction.oneVsRest == actualDigit) {
                    correctOneVsRest++;
                }
                if (prediction.oneVsOne == actualDigit) {
                    correctOneVsOne++;
                }
            }
            double size = dataSetB.size();
            System.out.println("\n--- " + label + " Success Rate ---");
            System.out.println("1-vs-Rest Correct: " + correctOneVsRest + " / " + dataSetB.size());
            System.out.println("1-vs-Rest Success Rate: " + (correctOneVsRest / size) * 100.0 + "%");
            System.out.println("1-vs-1 Correct: " + correctOneVsOne + " / " + dataSetB.size());
            System.out.println("1-vs-1 Success Rate: " + (correctOneVsOne / size) * 100.0 + "%");
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
            int predictedDigit = algorithm.predict(sample, dataSetA);
            if (actualDigit == predictedDigit) {
                correctMatches++;
            }
        }

        double successRate = (correctMatches / (double) dataSetB.size()) * 100.0;
        System.out.println("\n--- " + label + " Success Rate ---");
        System.out.println("Correct Matches: " + correctMatches + " / " + dataSetB.size());
        System.out.println("Success Rate: " + successRate + "%");
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
    private static void PrintDataUserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB) {
    	Scanner scanner = new Scanner(System.in); 
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
		        break;// Placeholder implementation
            // In a real scenario, this method would contain the logic for SVM prediction
		        
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
            System.out.println("0 -> Exit");
            System.out.print("\nEnter your choice (0-4): ");
            
            try {
                int choice = scanner.nextInt();
                
                switch (choice) {
                	case 1:
                		PrintDataUserInterface(dataSetA, dataSetB);
                		break;
                
                    case 2:
                    	evaluateAlgorithm(dataSetA, dataSetB, EUCLIDEAN_DISTANCE, "Euclidean Distance"); // tarin on A, test on B
                        break;
                        
                    case 3:
                    	evaluateAlgorithm(dataSetA, dataSetB, MULTI_LAYER_PERCEPTRON, "Multi Layer Perceptron"); // train on A, test on B
                    	break;

                    case 4:
                    	evaluateAlgorithm(dataSetB, dataSetA, DISTANCE_FROM_CENTROID, "Distance From Centroid"); // train on A, test on B
                    	break;
                    
                    case 5:
                        evaluateAlgorithm(dataSetA, dataSetB, SUPPORT_VECTOR_MACHINE, "Support Vector Machine"); // train on A, test on B
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