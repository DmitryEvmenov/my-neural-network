using System;
using MyNeuralNetwork.Enums;

namespace MyNeuralNetwork.Models
{
    class Neuron
    {
        private readonly NeuronTypes _type;

        public Neuron(double[] inputs, double[] weights, NeuronTypes type)
        {
            _type = type;
            Weights = weights;
            Inputs = inputs;
        }

        public double[] Weights { get; set; }

        public double[] Inputs { get; set; }

        public double Output => Activator(Inputs, Weights);

        private static double Activator(double[] i, double[] w)
        {
            double sum = 0;

            for (var l = 0; l < i.Length; ++l)
                sum += i[l] * w[l];

            return Math.Pow(1 + Math.Exp(-sum), -1);
        }

        public double Derivativator(double outsignal) => outsignal * (1 - outsignal);

        public double Gradientor(double error, double dif, double gSum) => _type == NeuronTypes.Output
            ? error * dif
            : gSum * dif;
    }
}
