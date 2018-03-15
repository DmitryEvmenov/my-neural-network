using System;
using MyNeuralNetwork.Enums;
using MyNeuralNetwork.Helpers;
using MyNeuralNetwork.Models;

namespace MyNeuralNetwork.WorkerServices
{
    class NetworkTrainer
    {
        private readonly double _allowedErrorRate;

        public NetworkTrainer(double allowedErrorRate = Network.AllowedErrorRate)
        {
            _allowedErrorRate = allowedErrorRate;
        }
        public Network Train(Network net)
        {
            Console.WriteLine("=======Training Started========");
            var iterationError = new double[net.InputLayer.Trainset.Length];
            double errorRate;
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
            } while (errorRate > _allowedErrorRate);

            net.HiddenLayer.InitWeights(MemoryModes.Set);
            net.OutputLayer.InitWeights(MemoryModes.Set);

            Console.WriteLine("========Training Ended========");

            return net;
        }
    }
}
