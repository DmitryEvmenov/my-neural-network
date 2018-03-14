using System;
using MyNeuralNetwork.Models;

namespace MyNeuralNetwork.WorkerServices
{
    class NetworkTester
    {
        public void Test(Network net)
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
