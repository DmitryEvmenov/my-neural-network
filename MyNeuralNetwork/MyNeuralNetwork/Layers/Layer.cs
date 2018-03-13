using System;
using System.Globalization;
using System.IO;
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
            NeuronType = neuronType;

            var weights = WeightInitialize(MemoryModes.Get);

            for (var i = 0; i < neuronsCount; ++i)
            {
                var tempWeights = new double[prevLayerNeuronsCount];
                for (var j = 0; j < prevLayerNeuronsCount; ++j)
                    tempWeights[j] = weights[i, j];

                Neurons[i] = new Neuron(null, tempWeights, neuronType);
            }
        }
        protected int NeuronsCount;
        protected int PrevLayerNeuronsCount;
        protected const double LearningRate = 0.1d;
        protected NeuronTypes NeuronType;

        public Neuron[] Neurons { get; set; }

        public double[] Data
        {
            set
            {
                foreach (var neuron in Neurons)
                    neuron.Inputs = value;
            }
        }

        public double[,] WeightInitialize(MemoryModes memoryMode)
        {
            var weights = new double[NeuronsCount, PrevLayerNeuronsCount];

            Console.WriteLine($"{NeuronType} layer weights are being initialized...");
            var memoryDoc = new XmlDocument();

            memoryDoc.Load(Path.Combine("../../../", Path.Combine("Memory", $"{NeuronType}_memory.xml")));
            var memoryEl = memoryDoc.DocumentElement;
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
            memoryDoc.Save($"{NeuronType}_memory.xml");
            Console.WriteLine($"{NeuronType} layer weights have been initialized...");
            return weights;
        }

        protected void AdjustWeights(double[] grSums)
        {
            for (var i = 0; i < NeuronsCount; ++i)
            {
                double error = 0;
                double gSum = 0;

                if (NeuronType == NeuronTypes.Hidden)
                {
                    gSum = grSums[i];
                }
                else if (NeuronType == NeuronTypes.Output)
                {
                    error = grSums[i];
                }

                for (var n = 0; n < PrevLayerNeuronsCount; ++n)
                {
                    Neurons[i].Weights[n] += LearningRate * Neurons[i].Inputs[n] *
                                             Neurons[i].Gradientor(error, Neurons[i].Derivativator(Neurons[i].Output),
                                                 gSum);
                }
            }
        }

        public abstract void Recognize(Network net, Layer nextLayer);
        public abstract double[] BackwardPass(double[] grSums);
    }
}
