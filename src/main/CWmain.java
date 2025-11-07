package main;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CWmain {

    private static final String FILE_PATH = "/home/casualstrategy/Downloads/dataSet1.csv";
    private static final int BIT_OF_DIGIT = 65;
    private static final int BITMAPS_TO_DISPLAY = 20;

    private static List<List<Integer>> readCsvFile(String filePath, int BIT_OF_DIGIT) {
        List<List<Integer>> dataSet = new ArrayList<>();
        int rowCount = 0;

        System.out.println("Starting to read CSV fildataSet.size()e: " + filePath);

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
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
            System.err.println("Ensure the file exists at the correct path: " + filePath);
            System.err.println("Details: " + e.getMessage());
            return null; 
        }
    }

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


	public static void main(String[] args) {
	    List<List<Integer>> dataSet = readCsvFile(FILE_PATH, BIT_OF_DIGIT);
	    
	    if (dataSet != null && !dataSet.isEmpty()) {
	        System.out.println("\n--- Data Processing Complete ---");
	        System.out.println("Total rows successfully loaded: " + dataSet.size());
	        
//	        printDataSet(dataSet); // prints entire data set
	        printLimitedDataSet(dataSet); // prints BITMAPS_TO_DISPLAY amount of bitmaps
	    } else {
	        System.out.println("\nNo data was loaded or an error occurred during reading.");
	    }
	}
}