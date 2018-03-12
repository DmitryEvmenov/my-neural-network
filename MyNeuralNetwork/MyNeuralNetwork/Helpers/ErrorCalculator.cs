using System;

namespace MyNeuralNetwork.Helpers
{
    static class ErrorCalculator
    {
        public static double CalcIterationError(double[] errors)
        {
            double sum = 0;
            foreach (var error in errors)
                sum += Math.Pow(error, 2);

            return 0.5d * sum;
        }

        public static double CalcRoundError(double[] iterationsErrors)
        {
            double sum = 0;

            foreach (var error in iterationsErrors)
                sum += error;

            return sum / iterationsErrors.Length;
        }
    }
}
