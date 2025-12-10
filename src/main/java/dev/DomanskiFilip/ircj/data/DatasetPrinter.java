package data;

import constants.Constants;
import java.util.List;

public class DatasetPrinter {

    public static void printDataSet(List<List<Integer>> dataSet) {
        System.out.println("--- Entire Dataset ---");
        for (int sampleIndex = 0; sampleIndex < dataSet.size(); sampleIndex++) {
            printRow(sampleIndex, dataSet.get(sampleIndex));
        }
    }

    public static void printLimitedDataSet(List<List<Integer>> dataSet) {
        System.out.println("--- First " + Constants.BITMAPS_TO_DISPLAY + " Samples ---");
        for (int sampleIndex = 0; sampleIndex < Math.min(Constants.BITMAPS_TO_DISPLAY, dataSet.size()); sampleIndex++) {
            printRow(sampleIndex, dataSet.get(sampleIndex));
        }
    }

    private static void printRow(int rowNumber, List<Integer> row) {
        int digitLabel = row.get(Constants.BITMAP_SIZE);

        System.out.print("Sample " + (rowNumber + 1) + " (Digit: " + digitLabel + "): [");
        for (int pixelIndex = 0; pixelIndex < Constants.BITMAP_SIZE; pixelIndex++) {
            System.out.print(row.get(pixelIndex));
            if (pixelIndex < Constants.BITMAP_SIZE - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
    }
}