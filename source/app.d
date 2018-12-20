import std.stdio;
import grain.probability.distribution;


struct HMC(alias distribution) if (isDistribution!distribution)
{
    import mir.random;
    import mir.random.variable;

    alias F = FloatOf!distribution;
    F position;
    F epsilon;
    size_t step;
    size_t maxIteration = 100;
    Random* gen;

    F hamiltonian(F position, F momentum)
    {
        return -distribution.logProb(position) + 0.5 * momentum * momentum;
    }

    void run()
    {
        import mir.math : fmin, exp;
        if (!this.gen)
        {
            this.gen = threadLocalPtr!Random;
        }
        auto rv = NormalVariable!F(0, 1);
        auto uv = UniformVariable!F(0, 1);

        auto momentum = rv(this.gen);

        // step
        foreach (t; 0 .. this.step)
        {
            writefln!"t=%d, p=%f, theta=%f"(t, momentum, this.position);
            auto momentum_half = momentum + this.epsilon * 0.5 * distribution.logProbGrad(this.position);
            auto new_position = this.position + this.epsilon * momentum_half;
            auto new_momentum = momentum_half + this.epsilon * 0.5 * distribution.logProbGrad(this.position);
            auto r = exp(hamiltonian(this.position, momentum) - hamiltonian(new_position, new_momentum));
            if (fmin(1.0, r) > uv(this.gen))
            {
                momentum = new_momentum;
                this.position = new_position;
            }
        }
    }
}



void testGammaHMC()
{
    auto dist = GammaDistribution!double(11, 13);
    HMC!dist sampler = {
        position: 0.1,
        epsilon: 0.05,
        step: 15
    };
    sampler.run();
}

void main()
{
    testGammaPlot();
    testGammaHMC();
    writeln("done");
}
