package main.java.dev.DomanskiFilip;

import algorithms.*;
import data.DatasetReader;
import evaluation.Evaluator;
import ui.UserInterface;

import java.util.List;

public class aiproject {

    private static final String DATASET_A_FILE_PATH = "src/main/java/dev/DomanskiFilip/aiproject/datasets/dataSetA.csv";
    private static final String DATASET_B_FILE_PATH = "src/main/java/dev/DomanskiFilip/aiproject/datasets/dataSetB.csv";

    public static void main(String[] args) {
        List<List<Integer>> dataSetA = DatasetReader.readCsvFile(DATASET_A_FILE_PATH);
        List<List<Integer>> dataSetB = DatasetReader.readCsvFile(DATASET_B_FILE_PATH);

        UserInterface.start(dataSetA, dataSetB);
        // Evaluator.runAllInOrder(dataSetA, dataSetB);
    }
}