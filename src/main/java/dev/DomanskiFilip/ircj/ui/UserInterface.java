package ui;

import algorithms.*;
import constants.Constants;
import data.DatasetPrinter;
import evaluation.Evaluator;
import features.FeatureMode;
import java.util.List;
import java.util.Scanner;

public class UserInterface {

    public static void start(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB) {
        Scanner scanner = new Scanner(System.in);
        boolean running = true;

        if (dataSetA == null || dataSetB == null) {
            System.err.println("Cannot run program: Dataset loading failed.");
            return;
        }

        while (running) {
            System.out.println("\n=== Actions: ===");
            System.out.println("1 -> Print Options");
            System.out.println("2 -> Euclidean Distance");
            System.out.println("3 -> Multi Layer Perceptron");
            System.out.println("4 -> Distance From Centroid");
            System.out.println("5 -> Support Vector Machine");
            System.out.println("6 -> K Nearest Neighbour");
            System.out.println("7 -> Mahalanobis Distance");
            System.out.println("8 -> All at Once");
            System.out.println("9 -> Run All Algorithms in Order");
            System.out.println("0 -> Exit");
            System.out.print("Choose between 0-9: ");

            try {
                int choice = scanner.nextInt();
                switch (choice) {
                    case 1:
                        printDataUserInterface(dataSetA, dataSetB, scanner);
                        break;

                    case 2:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new EuclideanDistance(), "Euclidean Distance");
                        System.out.println("Trained on B tested on A:");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new EuclideanDistance(), "Euclidean Distance");
                        break;

                    case 3:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(FeatureMode.RAW_ONLY), "MLP [Raw Only]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(FeatureMode.ALL), "MLP [All Features]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(FeatureMode.CENTROID_ONLY), "MLP [Centroid Only]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(FeatureMode.RAW_CENTROID), "MLP [Raw + Centroid]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(FeatureMode.RAW_KMEANS), "MLP [Raw + KMeans]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MultiLayerPerceptron(FeatureMode.RAW_GA), "MLP [Raw + GA]");

                        System.out.println("Trained on B tested on A:");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(FeatureMode.RAW_ONLY), "MLP [Raw Only]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(FeatureMode.ALL), "MLP [All Features]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(FeatureMode.CENTROID_ONLY), "MLP [Centroid Only]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(FeatureMode.RAW_CENTROID), "MLP [Raw + Centroid]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(FeatureMode.RAW_KMEANS), "MLP [Raw + KMeans]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MultiLayerPerceptron(FeatureMode.RAW_GA), "MLP [Raw + GA]");
                        break;

                    case 4:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new DistanceFromCentroid(), "Distance From Centroid");
                        System.out.println("Trained on B tested on A:");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new DistanceFromCentroid(), "Distance From Centroid");
                        break;

                    case 5:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(FeatureMode.CENTROID_ONLY), "SVM [Centroid Only]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(FeatureMode.ALL), "SVM [All Features]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(FeatureMode.RAW_CENTROID), "SVM [Raw + Centroid]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(FeatureMode.RAW_KMEANS), "SVM [Raw + KMeans]");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new SupportVectorMachine(FeatureMode.RAW_GA), "SVM [Raw + GA]");
                        
                        System.out.println("Trained on B tested on A:");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(FeatureMode.CENTROID_ONLY), "SVM [Centroid Only]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(FeatureMode.ALL), "SVM [All Features]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(FeatureMode.RAW_CENTROID), "SVM [Raw + Centroid]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(FeatureMode.RAW_KMEANS), "SVM [Raw + KMeans]");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new SupportVectorMachine(FeatureMode.RAW_GA), "SVM [Raw + GA]");
                        break;

                    case 6:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new KNearestNeighbour(), "K Nearest Neighbour");
                        System.out.println("Trained on B tested on A:");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new KNearestNeighbour(), "K Nearest Neighbour");
                        break;

                    case 7:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new MahalanobisDistance(), "Mahalanobis Distance");
                        System.out.println("Trained on B tested on A:");
                        Evaluator.evaluateAlgorithm(dataSetB, dataSetA, new MahalanobisDistance(), "Mahalanobis Distance");
                        break;

                    case 8:
                        System.out.println("Trained on A tested on B:");
                        Evaluator.evaluateAlgorithm(dataSetA, dataSetB, new AllAtOnce(), "All at Once");
                        break;
                    
                    case 9:
                        Evaluator.runAllInOrder(dataSetA, dataSetB);
                        break;

                    case 0:
                        System.out.println("\nExiting");
                        running = false;
                        break;

                    default:
                        System.out.println("\nInvalid choice. Please enter a number corresponding to available actions.");
                }

            } catch (Exception error) {
                System.out.println("\nInvalid input. Please enter a number corresponding to available actions.");
                scanner.nextLine();
            }
        }

        scanner.close();
    }

    private static void printDataUserInterface(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Scanner scanner) {
        boolean running = true;
        while (running) {
            System.out.println("\n=== Print Actions: ===");
            System.out.println("1 -> Print entire A");
            System.out.println("2 -> Print entire B");
            System.out.println("3 -> Print subset A (First " + Constants.BITMAPS_TO_DISPLAY + ")");
            System.out.println("4 -> Print subset B (First " + Constants.BITMAPS_TO_DISPLAY + ")");
            System.out.println("0 -> Exit");
            System.out.print("Choose between 0-4: ");
            try {
                int choice = scanner.nextInt();
                switch (choice) {
                    case 1:
                        if (dataSetA != null) {
                            DatasetPrinter.printDataSet(dataSetA);
                        }  
                        break;

                    case 2:
                        if (dataSetB != null) {
                            DatasetPrinter.printDataSet(dataSetB);
                        }   
                        break;

                    case 3:
                        if (dataSetA != null) {
                            DatasetPrinter.printLimitedDataSet(dataSetA);
                        }
                        break;

                    case 4:
                        if (dataSetB != null) {
                            DatasetPrinter.printLimitedDataSet(dataSetB);
                        }
                        break;

                    case 0:
                        running = false;
                        break;

                    default:
                        System.out.println("\nInvalid choice. Please enter a number corresponding to available actions.");
                }
            } catch (Exception e) {
                System.out.println("\nInvalid input. Please enter a number.");
                scanner.nextLine();
            }
        }
    }
}
