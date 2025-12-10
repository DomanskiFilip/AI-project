package algorithms;

public class PredictionResult {
    int predictedClass;
    double[] scores;

    PredictionResult(int predicted, double[] score) {
        this.predictedClass = predicted;
        this.scores = score;
    }
}