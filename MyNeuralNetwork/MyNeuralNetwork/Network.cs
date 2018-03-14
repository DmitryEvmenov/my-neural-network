﻿using System;
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
            Console.WriteLine("=======Training Started========");
            const double threshold = 0.001d;
            var tempMses = new double[4];
            double tempCost = 0;
            do
            {
                for (var i = 0; i < net._inputLayer.Trainset.Length; ++i)
                {
                    net.HiddenLayer.Data = net._inputLayer.Trainset[i].Item1.ToDoubles();
                    net.HiddenLayer.Recognize(null, net.OutputLayer);
                    net.OutputLayer.Recognize(net, null);

                    var errors = new double[net._inputLayer.Trainset[i].Item2.OpCount];
                    for (var x = 0; x < errors.Length; ++x)
                    {
                        errors[x] = net._inputLayer.Trainset[i].Item2[x] - net.Fact[x];
                    }

                    tempMses[i] = ErrorCalculator.CalcIterationError(errors);

                    var tempGsums = net.OutputLayer.BackwardPass(errors);
                    net.HiddenLayer.BackwardPass(tempGsums);
                }
                tempCost = ErrorCalculator.CalcRoundError(tempMses);

                Console.WriteLine($"Round error: {tempCost}");
            } while (tempCost > threshold);

            net.HiddenLayer.WeightInitialize(MemoryModes.Set);
            net.OutputLayer.WeightInitialize(MemoryModes.Set);

            Console.WriteLine("========Training Ended========");
        }

        static void Test(Network net)
        {
            Console.WriteLine("=============Testing===============");
            for (var i = 0; i < net._inputLayer.Trainset.Length; ++i)
            {
                net.HiddenLayer.Data = net._inputLayer.Trainset[i].Item1.ToDoubles();
                net.HiddenLayer.Recognize(null, net.OutputLayer);
                net.OutputLayer.Recognize(net, null);

                Console.WriteLine($"Expected outcome: {string.Join(", ", net._inputLayer.Trainset[i].Item2.ToDoubles())}");
                Console.WriteLine($"Actual outcome: {string.Join(", ", net.Fact)}");
                Console.WriteLine();
            }

            Console.WriteLine("=============Testing ended=============");
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
