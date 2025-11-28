package main;
import java.io.IOException;
import java.util.ArrayList;
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
        int predict(List<Integer> sample, List<List<Integer>> trainingSet); // sample -> row to predict, trainingSet -> dataset A (normally)
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
        @Override
        public int predict(List<Integer> sample, List<List<Integer>> trainingSet) {
            double[][] centroids = calculateCentroids(trainingSet);

            return 0; 
        }
    }

    // function to evaluate success rate of inputed algorithm
    private static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
         // If the provided algorithm is the MLP, print its configuration before running evaluation
        if (algorithm instanceof MultiLayerPerceptron) {
            System.out.println("\n--- " + label + " parameters used in calculation ---");
            System.out.println(((MultiLayerPerceptron) algorithm).getParameters());
        }
        
        // counting correct predictions train on A, test on B
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