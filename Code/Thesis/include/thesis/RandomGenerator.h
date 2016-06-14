/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <random>
#include <armadillo>

namespace ReLe
{
/*!
 * This class implements function to generate Random Number Generators.
 */
class RngGenerators
{
public:
    /*!
     * Constructor.
     * This function initializes the Random Number Generator of std and armadillo
     * libraries and the Random Number Generator of the class attribute with the provided seed.
     * \param seed the seed to be set
     */
    RngGenerators(unsigned int seed) : gen(seed)
    {
        srand(seed);
        arma::arma_rng::set_seed(seed);
    }

    /*!
     * Modifies the Random Number Generator of std and armadillo
     * libraries and the Random Number Generator of the class attribute with the provided seed.
     * \param seed the seed to be set
     */
    void seed(unsigned int seed)
    {
        srand(seed);
        arma::arma_rng::set_seed(seed);
        gen.seed(seed);
    }

    //! Random Number Generator variable based on Mersenne Twister Algorithm.
    std::mt19937 gen;
};

/*!
 * This class has some useful function to sample random numbers from
 * different probability distributions.
 */
class RandomGenerator
{
public:
    /*!
     * Generate a random unsigned integer number of 32 bits.
     * \return the random unsigned integer number
     */
    inline static uint32_t randu32()
    {
        return gen.gen();
    }

    /*!
     * Sample values from a normal distribution with mean = 0 and std = 1.
     * \return the random number sampled from the normal distribution
     */
    inline static double sampleNormal()
    {
        std::normal_distribution<> dist;
        return dist(gen.gen);
    }

    /*!
     * Sample values from a normal distribution with provided mean and standard deviation.
     * \param m the mean of the normal distribution
     * \param sigma the standard deviation of the normal distribution
     * \return the random number sampled from the normal distribution
     */
    inline static double sampleNormal(double m, double sigma)
    {
        std::normal_distribution<> dist(m, sigma);
        return dist(gen.gen);
    }

    inline static double sampleLogNormal()
    {
        std::lognormal_distribution<> dist;
        return dist(gen.gen);
    }

    inline static double sampleLogNormal(double m, double sigma)
    {
        std::lognormal_distribution<> dist(m, sigma);
        return dist(gen.gen);
    }

    /*!
     * Sample values from a uniform distribution with provided lower and higher values
     * where the higher one is excluded from the range.
     * \param lo the lower value of the uniform distribution
     * \param hi the upper value of the uniform distribution
     * \return the random number sampled from the uniform distribution
     */
    inline static double sampleUniform(const double lo, const double hi)
    {
        std::uniform_real_distribution<> dist(lo, hi);
        return dist(gen.gen);
    }

    /*!
     * Sample values from a uniform distribution with provided lower and higher values
     * where the lower one is excluded from the range.
     * \param lo the lower value of the uniform distribution
     * \param hi the upper value of the uniform distribution
     * \return the random number sampled from the uniform distribution
     */
    inline static double sampleUniformHigh(const double lo, const double hi)
    {
        std::uniform_real_distribution<> dist(-hi, -lo);
        return -dist(gen.gen);
    }

    /*!
     * Sample integer values from a uniform distribution with provided lower and higher values
     * where the lower one is excluded from the range.
     * \param lo the lower value of the uniform distribution
     * \param hi the upper value of the uniform distribution
     * \return the integer random number sampled from the uniform distribution
     */
    inline static std::size_t sampleUniformInt(const int lo, const int hi)
    {
        std::uniform_int_distribution<> dist(lo, hi);
        return dist(gen.gen);
    }

    /*!
     * Sample a random integer from 0 to n where n is the number of elements
     * of the given probability vector which indicates the probability of each
     * number to be sampled.
     * \param prob vector of probabilities
     * \return the sampled number
     */
    inline static std::size_t sampleDiscrete(std::vector<double>& prob)
    {
        std::discrete_distribution<std::size_t> dist(prob.begin(),
                prob.end());
        return dist(gen.gen);
    }

    /*!
     * Template function to sample a random integer from 0 to n where n is the number of elements
     * of the given probability vector (of arbitrary type) which indicates
     * the probability of each number to be sampled.
     * \param begin the first element of the probabilities vector
     * \param end the last element of the probabilities vector
     * \return the sampled number
     */
    template<class Iterator>
    inline static std::size_t sampleDiscrete(Iterator begin, Iterator end)
    {
        std::discrete_distribution<std::size_t> dist(begin, end);
        return dist(gen.gen);
    }

    /*!
     * Sample the outcome of an event (false or true) from a uniform distribution
     * with given probability.
     * \param prob the probability of the event
     * \return a bool value representing the happening of an event
     */
    inline static bool sampleEvent(double prob)
    {
        std::uniform_real_distribution<> dist(0, 1);
        return dist(gen.gen) < prob;
    }

    /*!
     * Set the seed of the Random Number Generator with a given seed.
     * \param seed the seed to be set
     */
    inline static void seed(unsigned int seed)
    {
        gen.seed(seed);
    }

private:
    //random generators
    static std::random_device rd;
    static RngGenerators gen;

};

}

#endif /* RANDOMGENERATOR_H_ */

