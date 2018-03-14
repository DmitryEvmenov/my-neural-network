using System;
using MyNeuralNetwork.Models;
using MyNeuralNetwork.WorkerServices;

namespace MyNeuralNetwork
{
    class Program
    {
        private static void Main(string[] args)
        {
            var net = new Network();
            var trainer = new NetworkTrainer();
            var tester = new NetworkTester();

            trainer.Train(net);
            tester.Test(net);

            Console.ReadKey();
        }
    }
}
