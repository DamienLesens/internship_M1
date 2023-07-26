#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/Algorithm.h"

#include "Config.h"
#include "Types.h"
#include "LowRankAlgo.h"
#include "algorithm/Connector.h"
//#include "util/Random.h"

#include <memory>
#include <utility>
#include <vector>
#include <random>

#define fl long double

class NeuronsExtraInfo;

/**
 * This class represents the implementation and adaptation of the Barnes�Hut algorithm. The parameters can be set on the fly.
 * In this instance, axons search for dendrites.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class LowRankApprox : public Algorithm {
public:
    LowRankApprox(int rank, LowRankAlgoEnum algo, int tot_number_neurons);

    /**
     * @brief computes the positive version of u
     */
    void compute_pos_u();

    /**
     * @brief computes the positive version of v
     */
    void compute_pos_v();

    int draw_one(std::vector<double> &intervals);

    /**
     * @brief searches for the index of the interval v falls into
     * @param partial
     * @param v
     * @return the index of the interval v belongs too
     */
    static int binary_search(std::vector<std::pair<double,int>> &partial,double v);
    /**
     * @brief draws nb times a number with probability proportional to the values in intervals
     * @param intervals
     * @param nb
     * @return a vector of the index drawn
     */
    static std::vector<int> draw_many(std::vector<double> &intervals,int nb);
    /**
     * @brief fills u and v with 1, the targets are thus chosen only depending on the number of vacant dendrites
     */
    void default_low_rank_approx();
    /**
     * @brief Adaptive Cross Approximation method, there is no guaranty that u and v will be positive
     */
    void aca_low_rank_approx();

    void nmf_low_rank_approx();

    void centroid_low_rank_approx();
    void centroid_weighed_low_rank_approx();

    /**
     * compute the error induced by the low rank approximation on the Kernel
     * selects rep lines and averages the L1 and Linf norm on scaled values
     */
    std::vector<fl> evaluate_error(int rep,bool print_res);
    std::vector<fl> evaluate_error_simple(int rep,bool print_res);

    double compute_approx(std::vector<std::vector<double>> &u,std::vector<std::vector<double>> &v, int i, int j);
    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses> update_connectivity(number_neurons_type number_neurons);
    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      Performs communication via MPI
     * @exception Can throw a RelearnException
     */
    void update_octree() {}

protected:
    LowRankAlgoEnum low_rank_algo;
    bool low_rank_approx_computed = false;

    std::vector<std::vector<double>> u,v;
    //positive counter-part of u and v
    std::vector<std::vector<double>> up,vp;

    std::vector<std::vector<int>> centroid;

    std::vector<int> choice_rank;

    int R;
    int N;
};
