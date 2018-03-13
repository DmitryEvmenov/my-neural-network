using System;
using MyNeuralNetwork.Enums;

namespace MyNeuralNetwork.Models
{
    class Neuron
    {
        private readonly NeuronTypes _type;
        private double[] _weights;
        private double[] _inputs;

        public Neuron(double[] inputs, double[] weights, NeuronTypes type)
        {
            _type = type;
            _weights = weights;
            _inputs = inputs;
        }

        public double[] Weights
        {
            get => _weights;
            set => _weights = value;
        }

        public double[] Inputs
        {
            get => _inputs;
            set => _inputs = value;
        }
        public double Output => Activator(_inputs, _weights);

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
