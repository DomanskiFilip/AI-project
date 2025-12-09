package algorithms;

import constants.Constants;
import java.util.List;
import java.util.PriorityQueue;

public class KNearestNeighbour implements Algorithm {
    private static final int K = 3;

    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        PriorityQueue<double[]> heap = new PriorityQueue<>((a, b) -> Double.compare(b[0], a[0]));

        for (List<Integer> candidate : trainingSet) {
            double distance = 0;
            for (int pixelIndex = 0; pixelIndex < Constants.BITMAP_SIZE; pixelIndex++) {
                double diff = sample.get(pixelIndex) - candidate.get(pixelIndex);
                distance += diff * diff;
            }

            if (heap.size() < K) {
                heap.offer(new double[] { distance, candidate.get(Constants.BITMAP_SIZE) });
            } else if (distance < heap.peek()[0]) {
                heap.poll();
                heap.offer(new double[] { distance, candidate.get(Constants.BITMAP_SIZE) });
            }
        }

        int[] votes = new int[10];
        while (!heap.isEmpty()) {
            int digit = (int) heap.poll()[1];
            votes[digit]++;
        }

        int bestDigit = 0;
        int bestVotes = 0;
        for (int digit = 0; digit < votes.length; digit++) {
            if (votes[digit] > bestVotes) {
                bestVotes = votes[digit];
                bestDigit = digit;
            }
        }
        return Integer.valueOf(bestDigit);
    }
}