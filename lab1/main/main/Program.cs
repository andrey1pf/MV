using System;
using System.IO;
using System.Diagnostics;

namespace main
{
    internal static class Program
    {
        private static StreamWriter sw = new StreamWriter("output.txt");
        private static StreamWriter resultSW = new StreamWriter("result.txt");
        private static StreamWriter matrixSW = new StreamWriter("matrix.txt");

        private static double GetRandomNumber(double minimum, double maximum)
        {
            Random random = new Random();
            double result = random.NextDouble() * (maximum - minimum) + minimum;

            while (Math.Abs(result - maximum) == 0)
            {
                result = random.NextDouble() * (maximum - minimum) + minimum;
            }
            return result;
        }

        private static void FillInTheUpperTriangleOfTheMatrix(float[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    if (i < j)
                    {
                        double randomNumber = GetRandomNumber(-2, 2);
                        float a = (float) Math.Pow(randomNumber, 5);
                        float b = (float) Math.Pow(randomNumber, 4);
                        if (a == 0 || b == 0) matrix[i, j] = 0;
                        else matrix[i, j] = a / b;
                    }
                }
            }
        }

        private static void MatrixDiagonalFilling(float[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                float sum = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    if (i != j)
                    {
                        sum += Math.Abs(matrix[i, j]);
                    }
                }

                matrix[i, i] = sum + 1;
            }
        }

        private static void FillingTheLowerTriangleOfTheMatrix(float[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                if (i != matrix.GetLength(0) - 1)
                {
                    for (int j = i + 1; j < matrix.GetLength(1); j++)
                    {
                        matrix[j, i] = matrix[i, j];
                    }
                }
            }
        }

        private static void TransposedMatrix(float[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    if (i < j)
                    {
                        (matrix[i, j], matrix[j, i]) = (matrix[j, i], matrix[i, j]);
                    }
                }
            }
        }

        private static void PrintMatrix(float[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sw.Write("\t{0}\t{1}", matrix[i, j], "|");
                }

                sw.WriteLine();
            }

            sw.WriteLine();
        }
        
        private static void PrintMatrix(float[,] matrix, int q)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrixSW.Write("\t{0}\t{1}", matrix[i, j], "|");
                }

                matrixSW.WriteLine();
            }

            matrixSW.WriteLine();
        }

        private static void VectorPadding(float[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                double randomNumber = GetRandomNumber(-2, 2);
                float a = (float) Math.Pow(randomNumber, 5);
                float b = (float) Math.Pow(randomNumber, 4);
                if (a == 0 || b == 0) vector[i] = 0;
                else vector[i] = a / b;
            }
        }

        private static void PrintVector(float[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                sw.Write("\t{0}\t{1}", vector[i], "|");
            }

            sw.WriteLine();
            sw.WriteLine();
        }

        private static void PrintVector(int[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                sw.Write("\t{0}\t{1}", vector[i], "|");
            }
            
            sw.WriteLine();
            sw.WriteLine();
        }

        private static void MultiplyingAVectorByAMatrix(float[] vector, float[,] matrix, float[] vectorB)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                float sum = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += vector[j] * matrix[i, j];
                }

                vectorB[i] = sum;
            }
        }

        private static void InverseMatrixByGaussJordanMethod(float[,] matrix, float[,] identityMatrix, int r)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    if (i == j) identityMatrix[i, j] = 1;
                    else identityMatrix[i, j] = 0;
                }
            }

            if (matrix[0, 0] == 0)
            {
                for (int i = 0; i < matrix.GetLength(0); ++i)
                {
                    if (matrix[i, 0] != 0)
                    {
                        SwapRows(matrix, i, 0);
                        SwapRows(identityMatrix, i, 0);
                        break;
                    }
                }
            }

            for (int i = 0; i < matrix.GetLength(0); ++i)
            {
                if (matrix[i, i] != 1)
                {
                    float delta = 1 / matrix[i, i];
                    for (int j = 0; j < matrix.GetLength(1); ++j)
                    {
                        matrix[i, j] *= delta;
                        identityMatrix[i, j] *= delta;
                    }

                    matrix[i, i] = 1;
                }

                NullingAColumn(matrix, identityMatrix, i);
                NullingUpAColumn(matrix, identityMatrix, i);
            }

            if (r == 0)
            {
                PrintMatrix(identityMatrix);
            }
        }

        private static void NullingAColumn(float[,] matrix, float[,] identityMatrix, int column)
        {
            for (int i = column + 1; i < matrix.GetLength(0); ++i)
            {
                float delta = -matrix[i, column];
                for (int j = 0; j < matrix.GetLength(1); ++j)
                {
                    matrix[i, j] += delta * matrix[column, j];
                    identityMatrix[i, j] += delta * identityMatrix[column, j];
                }
            }
        }

        private static void NullingUpAColumn(float[,] matrix, float[,] identityMatrix, int column)
        {
            for (int i = column - 1; i >= 0; --i)
            {
                float delta = -matrix[i, column];
                for (int j = 0; j < matrix.GetLength(1); ++j)
                {
                    matrix[i, j] += delta * matrix[column, j];
                    identityMatrix[i, j] += delta * identityMatrix[column, j];
                }
            }
        }

        private static void SwapRows(float[,] matrix, int rowA, int rowB)
        {
            int size = matrix.GetLength(0);

            float[] temp = new float[size];

            for (int i = 0; i < size; i++)
            {
                temp[i] = matrix[rowA, i];
            }

            for (int i = 0; i < size; i++)
            {
                matrix[rowA, i] = matrix[rowB, i];
            }

            for (int i = 0; i < size; i++)
            {
                matrix[rowB, i] = temp[i];
            }
        }

        private static void SwapRows(float[] vector, int rowA, int rowB)
        {
            (vector[rowA], vector[rowB]) = (vector[rowB], vector[rowA]);
        }

        private static void SwapRows(int[] vector, int rowA, int rowB)
        {
            (vector[rowA], vector[rowB]) = (vector[rowB], vector[rowA]);
        }

        private static float MatrixConditionNumber(float[,] matrix, float[,] inverseMatrix)
        {
            float normMatrix = GetCubNorm(matrix);
            float normInverseMatrix = GetCubNorm(inverseMatrix);
            float conditionNumber = normMatrix * normInverseMatrix;

            return conditionNumber;
        }

        private static float GetCubNorm(float[,] matrixCub)
        {
            float maxSum = 0;

            for (int i = 0; i < matrixCub.GetLength(0); i++)
            {
                float sum = 0;

                for (int j = 0; j < matrixCub.GetLength(0); j++)
                {
                    sum += Math.Abs(matrixCub[i, j]);
                }

                maxSum = Math.Max(maxSum, sum);
            }

            return maxSum;
        }
        
        private static float GetCubNorm(float[] vector)
        {
            float maxSum = 0;

            for (int i = 0; i < vector.GetLength(0); i++)
            {
                float sum = Math.Abs(vector[i]);

                maxSum = Math.Max(maxSum, sum);
            }

            return maxSum;
        }

        private static void GaussMethod(float[,] matrix, float[] vector, float[] result, int w)
        {
            for (int j = 0; j < matrix.GetLength(0); ++j)
            {
                float maxElement = 0;
                int maxRow = 0;
                for (int i = 0; i < matrix.GetLength(0); ++i)
                {
                    if (maxElement < Math.Abs(matrix[i, j]))
                    {
                        maxElement = Math.Abs(matrix[i, j]);
                        maxRow = i;
                    }
                }

                SwapRows(matrix, maxRow, j);
                SwapRows(vector, maxRow, j);

                for (int i = j; i < matrix.GetLength(0); i++)
                {
                    float temp = matrix[i, j];
                    for (int q = 0; q < matrix.GetLength(0); q++)
                    {
                        matrix[i, q] /= temp;
                    }

                    vector[i] /= temp;
                    if (i == j) continue;
                    for (int q = 0; q < matrix.GetLength(0); q++)
                    {
                        matrix[i, q] -= matrix[j, q];
                    }

                    vector[i] -= vector[j];
                }
            }

            for (int k = matrix.GetLength(0) - 1; k >= 0; k--)
            {
                result[k] = vector[k];
                for (int i = 0; i < k; i++)
                    vector[i] -= matrix[i, k] * result[k];
            }

            if (w == 0)
            {
                sw.WriteLine();
                sw.WriteLine("Полученная матрица:");
                PrintMatrix(matrix);
                
                sw.WriteLine("Полученный вектор:");
                PrintVector(result);
                sw.WriteLine();
            }
        }

        public static Stopwatch st4 = new Stopwatch();
        public static float[] timeLUP = new float[100];
        public static Stopwatch st5 = new Stopwatch();
        public static float[] timeLUPSolution = new float[100];
        private static int iterations = 0;
        private static void LUP(float[,] lMatrix, float[,] uMatrix, float[,] matrix, int[] p, float[] vectorB, int r)
        {
            st4.Start();
            int size = matrix.GetLength(0);
            float[,] matrixClone = new float[size, size];
            float[,] R = new float[size, size];
            matrixClone = (float[,]) matrix.Clone();
            uMatrix = (float[,]) matrixClone.Clone();

            for (int i = 0; i < size; ++i)
            {
                p[i] = i;
            }

            for (int k = 0; k < size; ++k)
            {
                float maxElement = 0;
                int maxRow = 0;
                for (int i = k; i < size; ++i)
                {
                    if (maxElement < Math.Abs(matrixClone[i, k]))
                    {
                        maxElement = Math.Abs(matrixClone[i, k]);
                        maxRow = i;
                    }
                }

                SwapRows(p, maxRow, k);
                SwapRows(matrixClone, maxRow, k);
                
            }
            for (int i = 0; i < size; i++)
            {
                for (int j = i; j < size; j++)
                {
                    lMatrix[j, i]=uMatrix[j, i]/uMatrix[i,i];
                }
            }
                
            for(int q = 1; q < size; q++)
            {
                for (int i = q - 1; i < size; i++)
                {
                    for (int j = i; j < size; j++)
                    {
                        lMatrix[j, i] = uMatrix[j, i] / uMatrix[i, i];
                    }
                }

                for (int i = q; i < size; i++)
                {
                    for (int j = q - 1; j < size; j++)
                    {
                        uMatrix[i, j] -= lMatrix[i, q - 1] * uMatrix[q - 1, j];
                    }
                }
            }

            st4.Stop();
            if (r == 0)
            {
                sw.WriteLine("Полученная матрица L:");
                PrintMatrix(lMatrix);
                sw.WriteLine("Полученная матрица U:");
                PrintMatrix(uMatrix);
                sw.WriteLine("Полученный вектор P:");
                PrintVector(p);
                LUPSolution(matrix, lMatrix, uMatrix, vectorB, p);
            }
            else
            {
                st5.Start();
                LUPSolution(matrix, lMatrix, uMatrix, vectorB, p);
                st5.Stop();
                timeLUP[iterations] = (float) st4.Elapsed.TotalSeconds;
                timeLUPSolution[iterations] = (float) st5.Elapsed.TotalSeconds;
                ++iterations;
            }

            //MatrixMultiplication(matrix, lMatrix, uMatrix, R, p, size); // Проверка
        }
        
        private static void LUPSolution(float[,] matrix, float[,] lmatrix, float[,] umatrix, float[] vectorB, int[] p)
        {
            var A = (float[,]) matrix.Clone();
            var l = (float[,]) lmatrix.Clone();
            var u = (float[,]) umatrix.Clone();
            var p1 = (int[]) p.Clone();
            int n = A.GetLength(0);
            float[] w = new float[n];
            float[] q = new float[n];
            float sum;

            for (int i = 0; i < n; ++i)
            {
                q[i] = vectorB[p1[i]];
            }

            for (int i = 0; i < n; ++i)
            {
                sum = 0;
                for (int j = 0; j < i; ++j)
                {
                    sum += l[i, j] * q[j];
                }

                q[i] -= sum;
            }

            for (int i = n - 1; i >= 0; --i)
            {
                sum = 0;
                for (int j = i + 1; j < n; ++j)
                {
                    sum += u[i, j] * w[j];
                }

                q[i] -= sum;
                w[i] = q[i] / u[i, i];
            }
        }

        private static void MatrixMultiplication(float[,] matrix, float[,] A, float[,] B, float[,] R, int[] pMatrix, int n)
        {
            float[,] C = new float[n, n];
            float[,] D = new float[n, n];
            for (int i = 0; i < n; ++i)
            {
                int index = pMatrix[i];
                for (int j = 0; j < n; ++j)
                {
                    C[i, j] = 0;
                    C[i, index] = 1;
                }
            }
            
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        D[i, j] += matrix[i, k] * C[k, j];
                    }
                    if (Math.Abs(D[i, j]) < 0.0000001)
                    {
                        D[i, j] = 0;
                    }
                }
            }
            
            //sw.WriteLine("AP: ");
            //PrintMatrix(D);
            
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        R[i, j] += A[i, k] * B[k, j];
                    }
                    if (Math.Abs(R[i, j]) < 0.0000001)
                    {
                        R[i, j] = 0;
                    }
                }
            }
            //sw.WriteLine("LU:");
            //PrintMatrix(R);
        }

        private static void MultiplicationMatrix(float[,] A, float[,] B, float[,] C)
        {
            int n = A.GetLength(0);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        C[i, j] += A[i, k] * B[k, j];
                    }
                    if (Math.Abs(C[i, j]) < 0.0000001)
                    {
                        C[i, j] = 0;
                    }
                }
            }
        }

        // for square matrix
        public static float[,] GetTransposedMatrix(float[,] A)
        {
            A = (float[,])A.Clone();

            int size = A.GetLength(0);

            float[,] Answer = new float[size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    Answer[i, j] = A[j, i];
                }
            }

            return Answer;
        }
        public static float[,] GetSquareRootOfMatrix(float[,] A)
        {
            A = (float[,])A.Clone();
            float[,] B = new float[,] {{0}};
            
            if (A[0, 0] < 0)
            {
                return null;
            }

            int size = A.GetLength(0);

            float[,] U = new float[size, size];

            U[0, 0] = (float)Math.Sqrt(A[0, 0]);

            for (int i = 1; i < size; i++)
            {
                U[0, i] = A[0, i] / U[0, 0];
            }

            for (int i = 1; i < size; i++)
            {
                float sum = 0;
                for (int k = 0; k < i; k++)
                {
                    sum +=(float)Math.Pow(U[k, i], 2);
                }
                if (A[i, i] - sum < 0)
                {
                    sw.WriteLine("Разница диагонального элемента и суммы меньше 0");
                    return B;
                }

                U[i, i] = (float)Math.Sqrt(A[i, i] - sum);

                for (int j = i+1; j < size; j++)
                {
                    float sum2 = 0;

                    for (int k = 0; k < i; k++)
                    {
                        sum2 += U[k, i] * U[k, j];
                    }

                    U[i, j] = (A[i, j] - sum2) / U[i, i];
                }

            }

            for (int j = 0; j < size; j++)
            {
                for (int i = j+1; i < size; i++)
                {
                    U[i, j] = 0;
                }
            }
            return U;
        }
        
        public static float[] SolveUpperTriangularMatrix(float[,] A, float[] B)
        {
            A = (float[,])A.Clone();
            B = (float[])B.Clone();

            int size = A.GetLength(0);

            float[] X = new float[size];

            for (int i = size - 1; i >= 0; i--)
            {
                float sum = 0;
                for (int j = size-1; j > i; j--)
                {
                    sum += X[j] * A[i, j];
                }
                sum = B[i] - sum;
                X[i] = sum / A[i, i];
            }
            return X;
        }

        public static float[] SolveLowerTriangularMatrix(float[,] A, float[] B)
        {
            A = (float[,])A.Clone();
            B = (float[])B.Clone();

            int size = A.GetLength(0);

            float[] X = new float[size];

            for (int i = 0; i < size; i++)
            {
                float sum = 0;
                for (int j = 0; j < i; j++)
                {
                    sum += X[j] * A[i, j];
                }
                sum = B[i] - sum;
                X[i] = sum / A[i, i];
            }
            return X;
        }

        private static void SquareRootMethod(float[,] SquareMatrix, float[] SquareVector, int q)
        {
            float[,] Ut = GetSquareRootOfMatrix(SquareMatrix);
            float[,] U = GetTransposedMatrix(Ut);
            float[] result = new float[SquareMatrix.GetLength(0)];
            result = SolveUpperTriangularMatrix(Ut, SolveLowerTriangularMatrix(U, SquareVector));
            PrintVector(result);
        }

        private static float[] SquareRootMethod(float[,] SquareMatrix, float[] SquareVector)
        {
            float[,] Ut = GetSquareRootOfMatrix(SquareMatrix);
            float[,] U = GetTransposedMatrix(Ut);
            float[] result = new float[SquareMatrix.GetLength(0)];
            result = SolveUpperTriangularMatrix(Ut, SolveLowerTriangularMatrix(U, SquareVector));
            return result;
        }

        private static void LDLT(float[,] matrix, int q)
        {
            int n = matrix.GetLength(0);
            var D = new float[n];
            var L = new float[n, n];
            var LT = new float[n, n];
            int i, j, k;
            double sum = 0;
            for (i = 0; i < n; i++)
            {
                for (j = i; j < n; j++)
                {
                    sum = matrix[j, i];
                    for (k = 0; k < i; k++)
                    {
                        sum -= L[i, k] * D[k] * L[j, k];
                    }

                    if (i == j)
                    {
                        D[i] = (float) sum;
                        L[i, i] = 1;
                        LT[i, i] = 1;
                    }
                    else
                    {
                        L[j, i] = (float) (sum / D[i]);
                        LT[i, j] = (float) (sum / D[i]);
                    }
                }
            }

            if (q == 1)
            {
                PrintMatrix(L);
                PrintVector(D);
                PrintMatrix(LT);   
            }
        }

        private static float VectorNorm(float[] vector)
        {
            int size = vector.GetLength(0);
            float max = 0;

            for(int i = 0; i<size; i++)
            {
                if (max < vector[i])
                {
                    max = vector[i];
                }
            }
            return max;
        }

        public static float[] relaxVector = new float[100];
        private static int RelaxationMethod(float[,] RelaxtionMatrix, float[] RelaxtionVector, float[] result, int p)
        {
            int n = RelaxtionMatrix.GetLength(0);
            float[] xNext = new float[n];
            float[] b = new float[n];
            float[] c = new float[n];
            float q = (float)(35) / (float)(40); // 1 - 5/40
            float delta = (float)0.01;
            int iteration = 0;

            MultiplyingAVectorByAMatrix(xNext, RelaxtionMatrix, c);
            for (int i = 0; i < n; ++i)
            {
                c[i] -= RelaxtionVector[i];
            }
            float norm = VectorNorm(c);

            while (norm > delta)
            {
                for (int i = 0; i < n; ++i)
                {
                    float sum = 0;
                    for (int w = 0; w < n; ++w)
                    {
                        if (w != i)
                        {
                            sum += RelaxtionMatrix[i, w] * xNext[w];
                        }
                    }

                    xNext[i] = (1 - q) * xNext[i] + (RelaxtionVector[i] - sum) * (q / RelaxtionMatrix[i, i]);
                }
            
                MultiplyingAVectorByAMatrix(xNext, RelaxtionMatrix, b);

                for (int i = 0; i < n; ++i)
                {
                    b[i] -= RelaxtionVector[i];
                }

                norm = VectorNorm(b);
                ++iteration;
            }

            if (p == 0)
            {
                PrintVector(xNext);
            }
            else
            {
                result = xNext;
            }

            return iteration;
        }

        private static void GetMatr(float[,] mas, float[,] p, int i, int j, int m)
        {
            int ki, kj, di, dj;
            di = 0;
            for (ki = 0; ki < m - 1; ++ki)
            {
                if (ki == i) di = 1;
                dj = 0;
                for (kj = 0; kj < m - 1; kj++)
                {
                    if (kj == j) dj = 1;
                    p[ki, kj] = mas[ki + di, kj + dj];
                }
            }
        }

        public static float Determinant(float[,] mas, int m)
        {
            int i, n = m - 1;
            float d = 0, k = 1;
            float[,] p = new float[m, m];

            if (m < 1) return 0;
            if (m == 1)
            {
                d = mas[0, 0];
                return d;
            }

            if (m == 2)
            {
                d = mas[0, 0] * mas[1, 1] - (mas[1, 0] * mas[0, 1]);
                return d;
            }

            if (m > 2)
            {
                for (i = 0; i < m; i++)
                {
                    GetMatr(mas, p, i, 0, m);
                    d += k * mas[i, 0] * Determinant(p, n);
                    k = -k;
                }
            }

            return d;
        }

        private static void ConvergenceProof(float[,] A, float[,] B, float q)
        {
            int n = A.GetLength(0);
            float detA = Determinant(A, n);
            float detB = Determinant(B, n);
            
            //sw.WriteLine("Определитель матрицы A: {0}", detA);
            //sw.WriteLine("Определитель матрицы B: {0}", detB);
            //sw.WriteLine("Параметр релаксации: {0}", q);
            if(Math.Abs(detA - detB) == 0 && detA != 0 && q > 0 && q < 2)
            {
                //sw.WriteLine("Метод релаксации сходится");
            }
        }

        private static float Average(float[] vec)
        {
            float sum = 0;
            for (int i = 0; i < vec.Length; ++i)
            {
                sum += vec[i];
            }

            return sum / vec.Length;
        }

        private static float[] vectorDifference(float[] A, float[] B)
        {
            float[] C = new float[A.Length];
            for (int i = 0; i < A.Length; ++i)
            {
                C[i] = A[i] - B[i];
            }

            return C;
        }
        
        private static float[] MinMaxAverage (float[] vec)
        {
            float[] result = new float[3];
            float sum = 0, min = vec[0], max = 0;
            int n = vec.Length;
            for (int i = 0; i < n; ++i)
            {
                if (vec[i] < min) min = vec[i];
                if (vec[i] > max) max = vec[i];
                sum += vec[i];
            }
            
            result[0] = min;
            result[1] = max;
            result[2] = sum / n;

            return result;
        }

        
        private static void Operation(float[,] matrix)
        {
            int matrixSize = matrix.GetLength(0);
            float[,] identityMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
            float[,] lMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
            float[,] uMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
            float[] vector = new float[matrixSize];
            float[] vectorB = new float[matrixSize];
            float[] result = new float[matrixSize];
            int[] pMatrix = new int[matrixSize];

            var transposedMatrixA = (float[,]) matrix.Clone();
            TransposedMatrix(transposedMatrixA);
            sw.WriteLine("Транспонированная матрица:");
            PrintMatrix(transposedMatrixA);

            VectorPadding(vector);
            sw.WriteLine("Вектор:");
            PrintVector(vector);

            MultiplyingAVectorByAMatrix(vector, matrix, vectorB);
            sw.WriteLine("Вектор после перемножения матрицы на вектор:");
            PrintVector(vectorB);

            sw.WriteLine("Обратная матрица:");
            var inverseMatrix = (float[,]) matrix.Clone();
            InverseMatrixByGaussJordanMethod(inverseMatrix, identityMatrix, 0);

            sw.WriteLine("Число обусловленности матрицы:");
            float a = MatrixConditionNumber(matrix, identityMatrix);
            sw.WriteLine(a);
            

            var GaussMatrix = (float[,]) matrix.Clone();
            var GaussVector = (float[]) vectorB.Clone();
            sw.WriteLine("Решение СЛАУ 𝐴𝑥 = 𝑏 методом Гаусса с выбором главного элемента по столбцу");
            GaussMethod(GaussMatrix, GaussVector, result, 0);

            var LUPVector = (float[]) vectorB.Clone();
            sw.WriteLine("LUP метод");
            LUP(lMatrix, uMatrix, matrix, pMatrix, LUPVector, 0);
            PrintVector(LUPVector);

            sw.WriteLine("Решить СЛАУ 𝐴𝑥 = 𝑏 методом квадратного корня: ");
            var SquareMatrix = (float[,]) matrix.Clone();
            var SquareVector = (float[]) vectorB.Clone();
            SquareRootMethod(SquareMatrix, SquareVector, 1);


            sw.WriteLine("LDL^T: ");
            var LDLTMatrix = (float[,]) matrix.Clone();
            LDLT(LDLTMatrix, 1);

            var RelaxationMatrix = (float[,]) matrix.Clone();
            var RelaxationVector = (float[]) vectorB.Clone();
            float[] u = new float[100];
            sw.WriteLine("Решение СЛАУ 𝐴𝑥 = 𝑏 методом релаксации (с параметром 1 − 5/40): ");
            int iterationCount = RelaxationMethod(RelaxationMatrix, RelaxationVector, u, 0);
            sw.WriteLine("Кол-во итераций: ");
            sw.WriteLine(iterationCount);
        }

        private static void TaskForMyMatrix()
        {
            float[,] myMatrixA = new float[,] {{40, 4, -1, -2}, {4, -40, -1, -4}, {-1, -1, 33, -5}, {-2, -4, -5, 35}};
            float[,] myMatrixB = new float[,]
            {
                {1, 6, 7, 8, 9, 10, 11, 12},
                {500, 5000, 50000, 500000, -5000, -50000, -500000, 1},
                {5, 4,3,2,1,0,-1,-2},
                {-995,-950,-500,4000,49000,-5,-4,-3},
                {-10,0,-1,-2,-3,-4,-5,-6},
                {-2014,2015,-2016,2017,-2018,2019,-2020,2021},
                {-1990,-1985,-1970,1935,-1860,10095,-10100,10105},
                {1010,1556,46304,491854,-53936, -41913, -507970,8293}
            };
            
            Operation(myMatrixA);
            Operation(myMatrixB);
        }
        
        static void Main()
        {
            int matrixSize = 256;
            int iteration = 100;
            float[] countConditionNumber = new float[iteration];
            float[] timeinverseMatrix = new float[iteration];
            float[] timeGauss = new float[iteration];
            float[] timeSquare = new float[iteration];
            float[] timeRelaxation = new float[iteration];
            float[] countIterationsRelaxation = new float[iteration];
            float[] gaussResult = new float[iteration];
            float[] squareVector = new float[iteration];
            float[] squareResult = new float[iteration];
            float[] relaxationResultVector = new float[iteration];
            float[] relaxResult = new float[iteration];
            float[] gaussMinMaxAverage = new float[3];
            float[] squareMinMaxAverage = new float[3];
            float[] relaxMinMaxAverage = new float[3];
            float min = 0, max = 0, minIteration = 0, maxIteration = 0;
            Stopwatch st = new Stopwatch();
            Stopwatch st1 = new Stopwatch();
            //Stopwatch st2 = new Stopwatch();
            Stopwatch st3 = new Stopwatch();
            Stopwatch st6 = new Stopwatch();

            for (int i = 0; i < iteration; ++i)
            {
                float[,] matrix = new float[matrixSize, matrixSize];
                float[,] identityMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
                float[,] lMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
                float[,] uMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
                float[] vector = new float[matrixSize];
                float[] vectorB = new float[matrixSize];
                float[] result = new float[matrixSize];
                float[] gaussresultVector = new float[matrixSize];
                int[] pMatrix = new int[matrixSize];
                FillInTheUpperTriangleOfTheMatrix(matrix);
                FillingTheLowerTriangleOfTheMatrix(matrix);
                MatrixDiagonalFilling(matrix);
                //sw.WriteLine("Матрица после заполнения:");
                //PrintMatrix(matrix);

                var transposedMatrixA = (float[,]) matrix.Clone();
                TransposedMatrix(transposedMatrixA);
                //sw.WriteLine("Транспонированная матрица:");
                //PrintMatrix(transposedMatrixA);

                VectorPadding(vector);
                //sw.WriteLine("Вектор:");
                //PrintVector(vector);

                MultiplyingAVectorByAMatrix(vector, matrix, vectorB);
                //sw.WriteLine("Вектор после перемножения матрицы на вектор:");
                //PrintVector(vectorB);

                //sw.WriteLine("Обратная матрица:");
                var inverseMatrix = (float[,]) matrix.Clone();
                st.Start();
                InverseMatrixByGaussJordanMethod(inverseMatrix, identityMatrix, 1);
                st.Stop();
                timeinverseMatrix[i] = (float) st.Elapsed.TotalSeconds;

                //sw.WriteLine("Число обусловленности матрицы:");
                float a = MatrixConditionNumber(matrix, identityMatrix);
                if (i == 0) min = a;
                countConditionNumber[i] = a;
                if (a > max)
                {
                    max = a;
                    PrintMatrix(matrix, 1);
                }
                else if (a < min) min = a;

                var GaussMatrix = (float[,]) matrix.Clone();
                var GaussVector = (float[]) vectorB.Clone();
                var GaussVectorq = (float[]) vectorB.Clone();
                //sw.WriteLine("Решение СЛАУ 𝐴𝑥 = 𝑏 методом Гаусса с выбором главного элемента по столбцу");
                st1.Start();
                GaussMethod(GaussMatrix, GaussVector, result, 1);
                gaussResult[i] = GetCubNorm(vectorDifference(result, GaussVectorq));
                st1.Stop();
                timeGauss[i] = (float) st1.Elapsed.TotalSeconds;
                

                //sw.WriteLine("LUP метод");
                //st2.Start();
                var LUPVector = (float[]) vectorB.Clone();
                LUP(lMatrix, uMatrix, matrix, pMatrix, LUPVector, 1);
                //st2.Stop();
                //timeLUP[i] = (float) st2.Elapsed.TotalSeconds;

                var SquareMatrix = (float[,]) matrix.Clone();
                var SquareVector = (float[]) vectorB.Clone();
                var SquareVectorq = (float[]) vectorB.Clone();
                st3.Start();
                squareVector = SquareRootMethod(SquareMatrix, SquareVector);
                st3.Stop();
                squareResult[i] = GetCubNorm(vectorDifference(squareVector, SquareVectorq));
                
                timeSquare[i] = (float) st3.Elapsed.TotalSeconds;

                var LDLTMatrix = (float[,]) matrix.Clone();
                LDLT(LDLTMatrix, 0);

                var RelaxationMatrix = (float[,]) matrix.Clone();
                var RelaxationVector = (float[]) vectorB.Clone();
                var RelaxationVectorq = (float[]) vectorB.Clone();
                
                //sw.WriteLine("Решение СЛАУ 𝐴𝑥 = 𝑏 методом релаксации (с параметром 1 − 5/40): ");
                st6.Start();
                int iterationCount = RelaxationMethod(RelaxationMatrix, RelaxationVector, relaxationResultVector, 1);
                st6.Stop();
                relaxResult[i] = GetCubNorm(vectorDifference(relaxationResultVector, RelaxationVectorq));
                timeRelaxation[i] = (float) st6.Elapsed.TotalSeconds;

                countIterationsRelaxation[i] = iterationCount;
                if (i == 0) minIteration = iterationCount;
                countConditionNumber[i] = iterationCount;
                if (iterationCount > maxIteration)
                {
                    maxIteration = iterationCount;
                    PrintMatrix(matrix, 1);
                }
                else if (iterationCount < minIteration) minIteration = iterationCount;
                
                Console.WriteLine(i);

                //var CholeskyMatrix = (float[,]) matrix.Clone();
                //var CholeskytransposedVector = (float[,]) transposedMatrixA.Clone();
                //sw.WriteLine("Критерий сходимости метода релаксации: ");
                //ConvergenceProof(CholeskyMatrix, CholeskytransposedVector, q); // хуита

                //PrintMatrix(matrix);
            }
            
            gaussMinMaxAverage = MinMaxAverage(gaussResult);
            squareMinMaxAverage = MinMaxAverage(squareResult);
            relaxMinMaxAverage = MinMaxAverage(relaxResult);

            resultSW.WriteLine("___________________Пункт 8___________________");
            
            resultSW.WriteLine("• Минимальное число обусловленности: " + min);
            resultSW.WriteLine("  Максимальное число обусловленности:  " + max);
            resultSW.WriteLine("  Среднее арифметическое число обусловленности: " + Average(countConditionNumber));
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Среднее время нахождения обратной матрицы: " + Average(timeinverseMatrix));
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Для Гаусса минимальное = " + gaussMinMaxAverage[0] + 
                               ", максимальное = " + gaussMinMaxAverage[1] + ", среднее = " + gaussMinMaxAverage[2]);
            resultSW.WriteLine("  Для квадратного корня минимальное = " + squareMinMaxAverage[0] + 
                               ", максимальное = " + squareMinMaxAverage[1] + ", среднее = " + squareMinMaxAverage[2]);
            resultSW.WriteLine("  Для метода релаксации = " + relaxMinMaxAverage[0] + 
                               ", максимальное = " + relaxMinMaxAverage[1] + ", среднее = " + relaxMinMaxAverage[2]);
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Среднее время решения СЛАУ методом Гаусса: " + Average(timeGauss));
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Среднее время построения 𝐿𝑈𝑃-разложения: " + Average(timeLUP));
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Среднее время решения СЛАУ 𝐿𝑈x: " + Average(timeLUPSolution));
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Среднее время решения СЛАУ методом квадратного корня: " + Average(timeSquare));
            resultSW.WriteLine("");
            
            resultSW.WriteLine("• Среднее время решения СЛАУ методом релаксации: " + Average(timeRelaxation));
            resultSW.WriteLine("");

            resultSW.WriteLine("• Максимальное количество итераций метода релаксации: " + maxIteration);
            resultSW.WriteLine("  Минимальное количество итераций метода релаксации: " + minIteration);
            resultSW.WriteLine("  Среднее количество итераций метода релаксации: " + Average(countIterationsRelaxation));
            resultSW.WriteLine("");
            
            TaskForMyMatrix();
            
            resultSW.Close();
            matrixSW.Close();
            //sw.Close();
        }
    }
}
