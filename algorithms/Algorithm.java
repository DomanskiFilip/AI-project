package algorithms;

import java.util.List;

// Interface for different classification algorithms
public interface Algorithm {
    Object predict(List<Integer> sample, List<List<Integer>> trainingSet);
}