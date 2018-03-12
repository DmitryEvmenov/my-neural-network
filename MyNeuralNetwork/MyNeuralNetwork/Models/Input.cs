namespace MyNeuralNetwork.Models
{
    class Input
    {
        public double Val1 { get; set; }
        public double Val2 { get; set; }

        public Input(double val1, double val2)
        {
            Val1 = val1;
            Val2 = val2;
        }

        public double[] ToDoubles() => new[] {Val1, Val2};

        public int ArgsCount = 2;
    }
}
