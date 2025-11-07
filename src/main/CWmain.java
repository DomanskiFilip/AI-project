package main;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CWmain {

    private static final String DATASET_A_FILE_PATH = "datasets/dataSetA.csv";
    private static final String DATASET_B_FILE_PATH = "datasets/dataSetB.csv";
    private static final int BIT_OF_DIGIT = 65;
    private static final int BITMAPS_TO_DISPLAY = 20;

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

    // Euclidean Distance
    private static void EuclideanDistance(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB) {
        int correctMatches = 0; // Count of correct matches
        int totalComparisons = dataSetA.size(); // Total rows in dataSetA

        for (int aIndex = 0; aIndex < dataSetA.size(); aIndex++) {
            List<Integer> selectedRow = dataSetA.get(aIndex);
            int actualDigit = selectedRow.get(BIT_OF_DIGIT - 1); // The 65th value (category)

            double minDistance = Double.MAX_VALUE;
            int closestRowIndex = -1;
            
            // Euclidian distance calculation
            for (int bIndex = 0; bIndex < dataSetB.size(); bIndex++) {
                List<Integer> currentRow = dataSetB.get(bIndex);

                double sum = 0.0;
                for (int j = 0; j < BIT_OF_DIGIT - 1; j++) { // Only compare the first 64 values
                    double diff = selectedRow.get(j) - currentRow.get(j);
                    sum += diff * diff;
                }
                double distance = Math.sqrt(sum);

                // update the closest row if the current distance is smaller
                if (distance < minDistance) {
                    minDistance = distance;
                    closestRowIndex = bIndex;
                }
            }

            // Check if the closest row has the same category (digit)
            int predictedDigit = dataSetB.get(closestRowIndex).get(BIT_OF_DIGIT - 1); // The 65th value (category)
            if (actualDigit == predictedDigit) {
                correctMatches++;
            }
        }

        // calculate and display the success rate
        double successRate = (correctMatches / (double) totalComparisons) * 100;
        System.out.println("\n--- Euclidean Distance Success Rate ---");
        System.out.println("Correct Matches: " + correctMatches + " / " + totalComparisons);
        System.out.println("Success Rate: " + successRate + "%");
    }
    
    // user interface 
    private static void UserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB) {
    	Scanner scanner = new Scanner(System.in); 
        boolean running = true;
        
        while (running) {
            System.out.println("\n=== Actions: ===");
            System.out.println("1 -> Print entire data set A");
            System.out.println("2 -> Print entire data set B");
            System.out.println("3 -> Print " + BITMAPS_TO_DISPLAY + " bitmaps from data set A");
            System.out.println("4 -> Print " + BITMAPS_TO_DISPLAY + " bitmaps from data set B");
            System.out.println("5 -> Get closest bitmap from data set B to selected bitmap from data set A useing Euclidean Distance");
            
            System.out.println("0 -> Exit");
            System.out.print("\nEnter your choice (0-5): ");
            
            try {
                int choice = scanner.nextInt();
                
                switch (choice) {
                    case 1:
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
                        
                    case 5:
                    	EuclideanDistance(dataSetA, dataSetB);
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
	    List<List<Integer>> dataSetA = readCsvFile(DATASET_A_FILE_PATH, BIT_OF_DIGIT);
	    List<List<Integer>> dataSetB = readCsvFile(DATASET_B_FILE_PATH, BIT_OF_DIGIT);
	    
	    UserInterface(dataSetA, dataSetB);
	    
	}
}