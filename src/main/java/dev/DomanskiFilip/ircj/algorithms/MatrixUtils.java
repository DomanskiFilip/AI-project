package algorithms;

// Utility class for matrix operations for Mahalanobis Distance
public class MatrixUtils {
    
    public static double[][] invertMatrix(double[][] matrix) {
        int size = matrix.length;
        double[][] augmented = new double[size][2 * size];

        for (int rowIndex = 0; rowIndex < size; rowIndex++) {
            System.arraycopy(matrix[rowIndex], 0, augmented[rowIndex], 0, size);
            augmented[rowIndex][rowIndex + size] = 1.0;
        }

        for (int colIndex = 0; colIndex < size; colIndex++) {
            int pivotRow = colIndex;
            double maxValue = Math.abs(augmented[pivotRow][colIndex]);
            for (int rowIndex = colIndex + 1; rowIndex < size; rowIndex++) {
                double value = Math.abs(augmented[rowIndex][colIndex]);
                if (value > maxValue) {
                    maxValue = value;
                    pivotRow = rowIndex;
                }
            }

            if (Math.abs(augmented[pivotRow][colIndex]) < 1e-9) {
                return null;
            }

            if (pivotRow != colIndex) {
                double[] temp = augmented[pivotRow];
                augmented[pivotRow] = augmented[colIndex];
                augmented[colIndex] = temp;
            }

            double pivotValue = augmented[colIndex][colIndex];
            for (int elementIndex = 0; elementIndex < 2 * size; elementIndex++) {
                augmented[colIndex][elementIndex] /= pivotValue;
            }

            for (int rowIndex = 0; rowIndex < size; rowIndex++) {
                if (rowIndex == colIndex) {
                    continue;
                }
                double factor = augmented[rowIndex][colIndex];
                for (int elementIndex = 0; elementIndex < 2 * size; elementIndex++) {
                    augmented[rowIndex][elementIndex] -= factor * augmented[colIndex][elementIndex];
                }
            }
        }

        double[][] inverse = new double[size][size];
        for (int rowIndex = 0; rowIndex < size; rowIndex++) {
            System.arraycopy(augmented[rowIndex], size, inverse[rowIndex], 0, size);
        }
        return inverse;
    }
}
