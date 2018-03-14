using System;
using MyNeuralNetwork.Enums;
using MyNeuralNetwork.Helpers;
using MyNeuralNetwork.Models;

namespace MyNeuralNetwork
{
    class Program
    {
        private static void Main(string[] args)
        {
            var net = new Network();

            Train(net);
            Test(net);

            Console.ReadKey();
        }

        static void Train(Network net)
        {
            Console.WriteLine("=======Training Started========");
            var iterationError = new double[4];
            double errorRate = 0;
            do
            {
                for (var i = 0; i < net.InputLayer.Trainset.Length; ++i)
                {
                    net.HiddenLayer.SetData(net.InputLayer.Trainset[i].Item1.ToDoubles());
                    net.HiddenLayer.Recognize(null, net.OutputLayer);
                    net.OutputLayer.Recognize(net, null);

                    var errors = new double[net.InputLayer.Trainset[i].Item2.OpCount];
                    for (var x = 0; x < errors.Length; ++x)
                    {
                        errors[x] = net.InputLayer.Trainset[i].Item2[x] - net.FactResult[x];
                    }

                    iterationError[i] = ErrorCalculator.CalcIterationError(errors);

                    var tempGsums = net.OutputLayer.BackwardPass(errors);
                    net.HiddenLayer.BackwardPass(tempGsums);
                }
                errorRate = ErrorCalculator.CalcRoundError(iterationError);

                Console.WriteLine($"Round error: {errorRate}");
            } while (errorRate > Network.AllowedErrorRate);

            net.HiddenLayer.WeightInitialize(MemoryModes.Set);
            net.OutputLayer.WeightInitialize(MemoryModes.Set);

            Console.WriteLine("========Training Ended========");
        }

        static void Test(Network net)
        {
            Console.WriteLine("=============Testing===============");
            for (var i = 0; i < net.InputLayer.Trainset.Length; ++i)
            {
                net.HiddenLayer.SetData(net.InputLayer.Trainset[i].Item1.ToDoubles());
                net.HiddenLayer.Recognize(null, net.OutputLayer);
                net.OutputLayer.Recognize(net, null);

                Console.WriteLine($"Expected outcome: {string.Join(", ", net.InputLayer.Trainset[i].Item2.ToDoubles())}");
                Console.WriteLine($"Actual outcome: {string.Join(", ", net.FactResult)}");
                Console.WriteLine();
            }

            Console.WriteLine("=============Testing ended=============");
        }
    }
}
