using MyNeuralNetwork.Enums;
using MyNeuralNetwork.Models;

namespace MyNeuralNetwork.Layers
{
    class OutputLayer : Layer
    {
        public OutputLayer(int neuronsCount, int prevLayerNeuronsCount) 
            : base(neuronsCount, prevLayerNeuronsCount, NeuronTypes.Output) { }

        public override void Recognize(Network net, Layer nextLayer)
        {
            for (var i = 0; i < Neurons.Length; ++i)
                net.Fact[i] = Neurons[i].Output;
        }

        public override double[] BackwardPass(double[] grSums)
        {
            var grSum = new double[PrevLayerNeuronsCount];

            for (var j = 0; j < grSum.Length; ++j)
            {
                double sum = 0;
                for (var k = 0; k < Neurons.Length; ++k)
                    sum += Neurons[k].Weights[j] * Neurons[k].Gradientor(grSums[k], Neurons[k].Derivativator(Neurons[k].Output), 0);
                grSum[j] = sum;
            }

            AdjustWeights(grSums);

            return grSum;
        }
    }
}
