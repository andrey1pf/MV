using System;

namespace main
{
    public class EigenvalueAndEigenvector
    {
        public static void MatrixVectorMultiplication(double[,] matrix, double[] vector, double[] result)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                result[i] = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    result[i] += matrix[i, j] * vector[j];
                }
            }
        }
        
        public static void Start(double[,] A)
        {
            
            int n = A.GetLength(0);
            double alpha = 0, alpha1 = 0;
            double[] eigenvalues = new double[n];
            double[] eigenvaluesNew = new double[n];
            double[] eigenvector = new double[n];
            double exp = 0.01, accuracy = 1;
            for(int i = 0; i < n; i++)
            {
                eigenvalues[i] = 1;
            }
            for(int i = 0; i < n; i++)
            {
                alpha += Math.Pow(eigenvalues[i], 2);
            }

            alpha = Math.Sqrt(alpha);
            
            for(int i = 0; i < n; i++)
            {
                eigenvaluesNew[i] = eigenvalues[i] / alpha;
            }
            MatrixVectorMultiplication(A, eigenvaluesNew, eigenvector);
            
            while(exp < accuracy)
            {
                alpha1 = alpha;
                alpha = 0;
                for(int i = 0; i < n; i++)
                {
                    alpha += Math.Pow(eigenvector[i], 2);
                }
                alpha = Math.Sqrt(alpha);
                for(int i = 0; i < n; i++)
                {
                    eigenvaluesNew[i] = eigenvector[i] / alpha;
                }
                accuracy = Math.Abs(alpha - alpha1)/alpha;
                MatrixVectorMultiplication(A, eigenvaluesNew, eigenvector);
            }
            Console.WriteLine(alpha);
            for(int i = 0; i < n; i++)
            {
                Console.WriteLine(eigenvaluesNew[i]);
            }
        }
    }
}