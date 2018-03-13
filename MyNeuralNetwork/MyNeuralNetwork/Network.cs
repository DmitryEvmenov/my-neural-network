using System;
using MyNeuralNetwork.Enums;
using MyNeuralNetwork.Helpers;
using MyNeuralNetwork.Layers;

namespace MyNeuralNetwork
{
    class Network
    {
        private readonly InputLayer _inputLayer = new InputLayer();

        public HiddenLayer HiddenLayer = new HiddenLayer(4, 2);
        public OutputLayer OutputLayer = new OutputLayer(2, 4);

        public double[] Fact = new double[2];
        

        static void Train(Network net)
        {
            const double threshold = 0.001d;
            double[] tempMses = new double[4];
            double tempCost = 0;
            do
            {
                for (var i = 0; i < net._inputLayer.Trainset.Length; ++i)
                {
                    net.HiddenLayer.Data = net._inputLayer.Trainset[i].Item1.ToDoubles();
                    net.HiddenLayer.Recognize(null, net.OutputLayer);
                    net.OutputLayer.Recognize(net, null);

                    double[] errors = new double[net._inputLayer.Trainset[i].Item2.OpCount];
                    for (var x = 0; x < errors.Length; ++x)
                    {
                        errors[x] = net._inputLayer.Trainset[i].Item2[x] - net.Fact[x];
                    }

                    tempMses[i] = ErrorCalculator.CalcIterationError(errors);

                    double[] tempGsums = net.OutputLayer.BackwardPass(errors);
                    net.HiddenLayer.BackwardPass(tempGsums);
                }
                tempCost = ErrorCalculator.CalcRoundError(tempMses);

                Console.WriteLine($"{tempCost}");
            } while (tempCost > threshold);

            net.HiddenLayer.WeightInitialize(MemoryModes.Set);
            net.OutputLayer.WeightInitialize(MemoryModes.Set);
        }

        static void Test(Network net)
        {
            for (int i = 0; i < net._inputLayer.Trainset.Length; ++i)
            {
                net.HiddenLayer.Data = net._inputLayer.Trainset[i].Item1.ToDoubles();
                net.HiddenLayer.Recognize(null, net.OutputLayer);
                net.OutputLayer.Recognize(net, null);

                for (int j = 0; j < net.Fact.Length; ++j)
                    Console.WriteLine($"{net.Fact[j]}");

                Console.WriteLine();
            }
        }

        static void Main(string[] args)
        {
            var net = new Network();

            Train(net);
            Test(net);

            Console.ReadKey();
        }
    }
}
