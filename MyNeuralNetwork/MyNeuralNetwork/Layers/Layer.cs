using System;
using System.Globalization;
using System.Xml;
using MyNeuralNetwork.Enums;
using MyNeuralNetwork.Models;

namespace MyNeuralNetwork.Layers
{
    abstract class Layer
    {
        protected Layer(int neuronsCount, int prevLayerNeuronsCount, NeuronTypes neuronType)
        {
            NeuronsCount = neuronsCount;
            PrevLayerNeuronsCount = prevLayerNeuronsCount;
            Neurons = new Neuron[neuronsCount];

            double[,] weights = WeightInitialize(MemoryModes.Get, neuronType);

            for (int i = 0; i < neuronsCount; ++i)
            {
                double[] tempWeights = new double[prevLayerNeuronsCount];
                for (int j = 0; j < prevLayerNeuronsCount; ++j)
                    tempWeights[j] = weights[i, j];

                Neurons[i] = new Neuron(null, tempWeights, neuronType);
            }
        }
        protected int NeuronsCount;
        protected int PrevLayerNeuronsCount;
        protected const double LearningRate = 0.1d;

        public Neuron[] Neurons { get; set; }

        public double[] Data
        {
            set
            {
                foreach (var neuron in Neurons)
                    neuron.Inputs = value;
            }
        }

        public double[,] WeightInitialize(MemoryModes memoryMode, NeuronTypes type)
        {
            double[,] weights = new double[NeuronsCount, PrevLayerNeuronsCount];

            Console.WriteLine($"{type} weights are being initialized...");
            XmlDocument memoryDoc = new XmlDocument();

            memoryDoc.Load($"{type}_memory.xml");
            XmlElement memoryEl = memoryDoc.DocumentElement;
            switch (memoryMode)
            {
                case MemoryModes.Get:
                    for (var l = 0; l < weights.GetLength(0); ++l)
                        for (var k = 0; k < weights.GetLength(1); ++k)
                            weights[l, k] = double.Parse(memoryEl.ChildNodes.Item(k + weights.GetLength(1) * l).InnerText.Replace(',', '.'), CultureInfo.InvariantCulture);
                    break;
                case MemoryModes.Set:
                    for (var l = 0; l < Neurons.Length; ++l)
                        for (var k = 0; k < PrevLayerNeuronsCount; ++k)
                            memoryEl.ChildNodes.Item(k + PrevLayerNeuronsCount * l).InnerText = Neurons[l].Weights[k].ToString(CultureInfo.InvariantCulture);
                    break;
            }
            memoryDoc.Save($"{type}_memory.xml");
            Console.WriteLine($"{type} weights have been initialized...");
            return weights;
        }

        //abstract public void Recognize(Network net, Layer nextLayer);
        public abstract double[] BackwardPass(double[] stuff);
    }
}
