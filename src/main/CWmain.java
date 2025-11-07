package main;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CWmain {

    private static final String DATASET_A_FILE_PATH = "datasets/dataSetA.csv";
    private static final String DATASET_B_FILE_PATH = "datasets/dataSetB.csv";
    private static final int BIT_OF_DIGIT = 65;
    private static final int BITMAPS_TO_DISPLAY = 20;

    private static List<List<Integer>> readCsvFile(String DATASET_A_FILE_PATH, int BIT_OF_DIGIT) {
        List<List<Integer>> dataSet = new ArrayList<>();
        int rowCount = 0;

        System.out.println("Starting to read CSV fildataSet.size()e: " + DATASET_A_FILE_PATH);

        try (BufferedReader br = new BufferedReader(new FileReader(DATASET_A_FILE_PATH))) {
            String line;
            
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                
                if (values.length != BIT_OF_DIGIT) {
                    System.err.println("Warning: Row " + (rowCount + 1) + " has " + values.length + " columns, expected " + BIT_OF_DIGIT + ". Skipping row.");
                    continue;
                }
                
                List<Integer> currentRow = new ArrayList<>();
                boolean conversionError = false;
                
                for (String value : values) {
                    try {
                        currentRow.add(Integer.parseInt(value.trim()));
                    } catch (NumberFormatException e) {
                        System.err.println("Error: Could not convert value '" + value + "' to integer in row " + (rowCount + 1) + ". Skipping row.");
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

        } catch (IOException e) {
            System.err.println("\n--- ERROR: Failed to read the file ---");
            System.err.println("Ensure the file exists at the correct path: " + DATASET_A_FILE_PATH);
            System.err.println("Details: " + e.getMessage());
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

    private static void UserInterface(List<List<Integer>> dataSetA ,List<List<Integer>> dataSetB) {
    	Scanner scanner = new Scanner(System.in);
        boolean running = true;
        
        while (running) {
            System.out.println("\n=== Actions: ===");
            System.out.println("1 -> Print entire data set A");
            System.out.println("2 -> Print entire data set B");
            System.out.println("3 -> Print " + BITMAPS_TO_DISPLAY + " bitmaps from data set A");
            System.out.println("4 -> Print " + BITMAPS_TO_DISPLAY + " bitmaps from data set B");
            System.out.println("5 -> Exit");
            System.out.print("\nEnter your choice (1-5): ");
            
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
                        System.out.println("\nExiting");
                        running = false;
                        break;
                        
                    default:
                        System.out.println("\nInvalid choice. Please enter a number corresponting to avaliable actions.");
                }
                
            } catch (Exception e) {
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