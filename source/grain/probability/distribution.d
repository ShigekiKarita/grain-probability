module grain.probability.distribution;


import std.traits : ReturnType, isFloatingPoint;


enum bool isDistribution(alias d) = isFloatingPoint!(ReturnType!(d.logProb));


template FloatOf(alias d)
{
    static assert(isDistribution!d);
    alias FloatOf = ReturnType!(d.logProb);
}


struct GammaDistribution(F = float)
{
    import mir.math : exp, pow, log;
    import std.mathspecial : logGamma;

    const F concentration;
    const F rate;
    const F bias;

    this(F concentration, F rate = 1)
    {
        this.concentration = concentration;
        this.rate = rate;
        this.bias = this.concentration * this.rate.log - this.concentration.logGamma;
    }

    F logProb(F value)
    {
        return (this.concentration - 1) * value.log - this.rate * value + this.bias;
    }

    // TODO use grain-style backward
    F logProbGrad(F value)
    {
        return (this.concentration - 1) - this.rate;
    }
}


void testGammaPlot()
{
    import std.format : format;
    import std.algorithm : map;
    import std.typecons : tuple;
    import ggplotd.ggplotd : GGPlotD, putIn;
    import ggplotd.geom : geomLine;
    import ggplotd.stat : statFunction;
    import ggplotd.range : mergeRange;
    import ggplotd.legend : discreteLegend;
    import ggplotd.aes : merge;
    import mir.math : exp;

    auto gg = GGPlotD();
    foreach (k; 1 .. 10)
    {
        auto f = GammaDistribution!double(k, 1);
        auto g = (double x) => exp(f.logProb(x));
        auto xs = statFunction(g, 0, 10.0, 100)
            // FIXME: find better way to provide labels instead of this
            .map!(a => a.merge(tuple!"colour"("%d".format(k))));
        gg = mergeRange(tuple!"colour"(k), xs)
            .geomLine
            .putIn(gg);
    }
    gg = gg.put(discreteLegend);
    gg.save("gamma.png");
}
