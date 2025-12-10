package data;

import constants.Constants;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DatasetReader {

    public static List<List<Integer>> readCsvFile(String dataSetFilePath) {
        List<List<Integer>> dataSet = new ArrayList<>();
        System.out.println("Reading CSV: " + dataSetFilePath);
        try (Scanner scanner = new Scanner(new File(dataSetFilePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] values = line.split(",");

                if (values.length != Constants.ENTIRE_BITMAP_SIZE) {
                    continue;
                }

                List<Integer> currentRow = new ArrayList<>();
                boolean error = false;

                for (int columnIndex = 0; columnIndex < Constants.ENTIRE_BITMAP_SIZE; columnIndex++) {
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
}