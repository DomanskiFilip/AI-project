package main;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CWmain {

    private static final String DATASET_A_FILE_PATH = "datasets/dataSetA.csv";
    private static final String DATASET_B_FILE_PATH = "datasets/dataSetB.csv";
    private static final int BIT_OF_DIGIT = 65; // 64 bits + 1 category
    private static final int BITMAPS_TO_DISPLAY = 20;


    // placeholder Algorythm interface
    public interface Algorithm {
        int predict(List<Integer> sample, List<List<Integer>> referenceSet, int bitOfDigit);
    }

    private static final Algorithm EUCLIDEAN_DISTANCE = new EuclideanDistance();
    private static final Algorithm MULTI_LAYER_PERCEPTRON = new MultiLayerPerceptron();

    // Euclidean Distance Algorythm
    private static class EuclideanDistance implements Algorithm {

        @Override
        public int predict(List<Integer> sample, List<List<Integer>> referenceSet, int bitOfDigit) {
            if (referenceSet == null || referenceSet.isEmpty()) {
                throw new IllegalArgumentException("Reference dataset must not be null or empty.");
            }

            double minDistance = Double.MAX_VALUE;
            List<Integer> closest = null;

            for (int c = 0; c < referenceSet.size(); c++) {
                List<Integer> candidate = referenceSet.get(c);
                double sum = 0.0;
                for (int i = 0; i < bitOfDigit - 1; i++) {
                    double distance = sample.get(i) - candidate.get(i);
                    sum += distance * distance;
                }
                if (sum < minDistance) {
                    minDistance = sum;
                    closest = candidate;
                }
            }

            return closest != null ? closest.get(bitOfDigit - 1) : -1;
        }
    }
    
    // Multi Layer Perceptron Algorythm
    private static class MultiLayerPerceptron implements Algorithm {
        @Override
        public int predict(List<Integer> sample, List<List<Integer>> referenceSet, int bitOfDigit) {
            
        }
    }

    // function to evaluate success rate of inputed algorithm
    private static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        int correctMatches = 0;
        for (int s = 0; s < dataSetA.size(); s++) {
            List<Integer> sample = dataSetA.get(s);
            int actualDigit = sample.get(BIT_OF_DIGIT - 1);
            int predictedDigit = algorithm.predict(sample, dataSetB, BIT_OF_DIGIT);
            if (actualDigit == predictedDigit) {
                correctMatches++;
            }
        }

        double successRate = (correctMatches / (double) dataSetA.size()) * 100.0;
        System.out.println("\n--- " + label + " Success Rate ---");
        System.out.println("Correct Matches: " + correctMatches + " / " + dataSetA.size());
        System.out.println("Success Rate: " + successRate + "%");
    }

    // function to read the csv files 
    private static List<List<Integer>> readCsvFile(String DATASET_A_FILE_PATH, int BIT_OF_DIGIT) {
        List<List<Integer>> dataSet = new ArrayList<>();
        int rowCount = 0;

        System.out.println("Starting to read CSV file: " + DATASET_A_FILE_PATH);

        try (Scanner scanner = new Scanner(new java.io.File(DATASET_A_FILE_PATH))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] values = line.split(",");

                // Ensure the row has the expected number of columns
                if (values.length != BIT_OF_DIGIT) {
                    System.err.println("Warning: Row " + (rowCount + 1) + " has " + values.length + " columns, expected " + BIT_OF_DIGIT + ". Skipping row.");
                    continue;
                }

                List<Integer> currentRow = new ArrayList<>();
                boolean conversionError = false;

                for (int i = 0; i < BIT_OF_DIGIT; i++) {
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
            System.err.println("Ensure the file exists at the correct path: " + DATASET_A_FILE_PATH);
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
            System.out.println("3 -> Multi Layer Perceptron");            System.out.println("0 -> Exit");
            System.out.print("\nEnter your choice (0-3): ");
            
            try {
                int choice = scanner.nextInt();
                
                switch (choice) {
                	case 1:
                		PrintDataUserInterface(dataSetA, dataSetB);
                		break;
                
                    case 2:
                    	evaluateAlgorithm(dataSetA, dataSetB, EUCLIDEAN_DISTANCE, "Euclidean Distance");
                        break;
                        
                    case 3:
                    	evaluateAlgorithm(dataSetA, dataSetB, MULTI_LAYER_PERCEPTRON, "Multi Layer Perceptron");
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
	    List<List<Integer>> dataSetA = readCsvFile(DATASET_A_FILE_PATH, BIT_OF_DIGIT);
	    List<List<Integer>> dataSetB = readCsvFile(DATASET_B_FILE_PATH, BIT_OF_DIGIT);
	    
        // start user interface
	    UserInterface(dataSetA, dataSetB);
	}
}