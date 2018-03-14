using MyNeuralNetwork.Models;

namespace MyNeuralNetwork.Layers
{
    class InputLayer
    {
        public (Input, Output)[] Trainset { get; } =
        {
            (new Input(0, 0), new Output(0, 1)),
            (new Input(0, 1), new Output(1, 0)),
            (new Input(1, 0), new Output(1, 0)),
            (new Input(1, 1), new Output(0, 1))
        };
    }
}
