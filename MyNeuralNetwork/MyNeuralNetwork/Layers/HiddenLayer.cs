using MyNeuralNetwork.Enums;

namespace MyNeuralNetwork.Layers
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int neuronsCount, int prevLayerNeuronsCount) 
            : base(neuronsCount, prevLayerNeuronsCount, NeuronTypes.Hidden) { }

        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hiddenOut = new double[Neurons.Length];

            for (int i = 0; i < Neurons.Length; ++i)
                hiddenOut[i] = Neurons[i].Output;

            nextLayer.Data = hiddenOut;
        }

        public override double[] BackwardPass(double[] grSums)
        {
            double[] grSum = null;

            AdjustWeights(grSums);

            return grSum;
        }
    }
}
