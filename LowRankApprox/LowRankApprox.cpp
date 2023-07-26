//
// Created by damien on 15/06/23.
//

#include "LowRankApprox.h"

#define REP(i,a,b) for (int i = (a); i <= (b); ++i)
#define REPD(i,a,b) for (int i = (a); i >= (b); --i)
#define FORI(i,n) REP(i,1,n)
#define FOR(i,n) REP(i,0,int(n)-1)
#define pb push_back
#include"LowRankAlgo.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "util/Timers.h"




LowRankApprox::LowRankApprox(int rank, LowRankAlgoEnum algo, int tot_number_neurons) {
    R = rank;
    N = tot_number_neurons;
    low_rank_algo = algo;
    LogFiles::print_message_rank(MPIRank::root_rank(), "rank : {}, algo : {}, N : {}",R, stringify(algo),N);
}

void LowRankApprox::compute_pos_u() {
    if (low_rank_algo == LowRankAlgoEnum::ACA) {
        up = std::vector<std::vector<double>>(R,std::vector<double>(N,0));
        std::vector<double> volume(R,0);
        std::vector<int> nb_zero(R,0);
        for(int k=0; k<R; k++) {
            for (int i=0; i<N; i++) {
                if (u[k][i]>=0) {
                    up[k][i]=u[k][i];
                }
                else {
                    nb_zero[k]+=1;
                }
                volume[k] += up[k][i];
            }
        }
    }
    else {
        up = u;
    }
}

void LowRankApprox::compute_pos_v() {
    if (low_rank_algo == LowRankAlgoEnum::ACA) {
        vp = std::vector<std::vector<double>>(R,std::vector<double>(N,0));
        std::vector<double> volume(R,0);
        std::vector<int> nb_zero(R,0);
        for(int k=0; k<R; k++) {
            for (int i=0; i<N; i++) {
                if (v[k][i]>=0) {
                    vp[k][i]=v[k][i];
                }
                else {
                    nb_zero[k]+=1;
                }
                volume[k] += vp[k][i];
            }
        }
    }
    else {
        vp = v;
    }
}

int LowRankApprox::binary_search(std::vector<std::pair<double,int>> &partial,double v) {
    int const K = partial.size();
    int bot=0;
    int top=K-1;
    int mid;
    while (top-bot > 1) {
        mid = (top+bot)/2;
        if (v < partial[mid].first)  {
            top = mid;
        }
        else {
            bot = mid;
        }
    }
    if (v<partial[bot].first) {
        return partial[bot].second;
    }
    else {
        return partial[top].second;
    }
}

std::vector<int> LowRankApprox::draw_many(std::vector<double> &intervals,int nb) {
    // be careful : some intervals might be of length 0
    // draw nb values following the uniform law on the intervals, costs O(max(K,nb*log(K))
    int const K = intervals.size();


    double sum = 0;
    //we remove the null intervals, so we have to store the index of each interval
    std::vector<std::pair<double,int>> partial(0);
    for(int i=0; i<K; i++) {
        if (intervals[i]>0) {
            if (partial.empty()) {
                partial.push_back(std::make_pair(intervals[i],i));
            }
            else {
                partial.push_back(std::make_pair(intervals[i]+partial.back().first,i));
            }
        }
        sum += intervals[i];

    }

    if (partial.empty()) {
        //all the intervals have length 0, so we return an empty list
        return {};
    }

    double v;
    std::vector<int> list={};
    for(int i=0; i<nb; i++) {
        v = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm,0,sum);
        list.push_back(binary_search(partial,v));
    }
    return list;
}

void print_vect(std::vector<fl> &v, std::string msg) {
    //std::cout << msg << std::endl;
    for(fl f : v) {
        std::cout<< f << ",";
    }
    std::cout << std::endl;
}

[[nodiscard]] std::tuple<PlasticLocalSynapses, PlasticDistantInSynapses, PlasticDistantOutSynapses> LowRankApprox::update_connectivity(number_neurons_type number_neurons) {

    //compute the low rank approx if not already done
    if (!low_rank_approx_computed) {
        //compute u and v, we need N

        LogFiles::print_message_rank(MPIRank::root_rank(), "computing low rank approximation");
        Timers::start(TimerRegion::COMPUTE_LOW_RANK_APPROX);

        switch (low_rank_algo) {
            case LowRankAlgoEnum::Default :
                default_low_rank_approx();
                break;
            case LowRankAlgoEnum::ACA :
                // O(R²*N)
                aca_low_rank_approx();
                break;
            case LowRankAlgoEnum::NMF :
                RelearnException::fail("Algorithm NMF not implemented yet");
                break;
            case LowRankAlgoEnum::Centroid :
                centroid_low_rank_approx();
                break;
            case LowRankAlgoEnum::Centroid_weighted :
                centroid_weighed_low_rank_approx();
                break;
        }
        LogFiles::print_message_rank(MPIRank::root_rank(), "putting to zero negative coefficients");
        //compute the positive versions of u and v, O(R*N)
        compute_pos_u();
        compute_pos_v();
        low_rank_approx_computed = true;
        Timers::stop_and_add(TimerRegion::COMPUTE_LOW_RANK_APPROX);


        Timers::start(TimerRegion::EVALUATE_ERROR);
        LogFiles::print_message_rank(MPIRank::root_rank(), "Evaluating the error");
        //O(R*N)
        //evaluate_error(100,true);
        Timers::stop_and_add(TimerRegion::EVALUATE_ERROR);

        choice_rank = std::vector<int>(R,0);




    }

    //lists containing neuron ids, an axon is in the list i if it wants to connect through the ith rank
    std::vector<std::vector<NeuronID>> need_excit(R, std::vector<NeuronID>(0));
    std::vector<std::vector<NeuronID>> need_inhib(R, std::vector<NeuronID>(0));

    if (low_rank_algo== LowRankAlgoEnum::Centroid) {
        //O(N)
        for(int k=0; k<R; k++) {

            for (int neuron_id: centroid[k]) {
                const NeuronID id{neuron_id};

                const auto number_vacant_axons = axons->get_free_elements(id);
                if (number_vacant_axons == 0) {
                    continue;
                }

                const auto dendrite_type_needed = axons->get_signal_type(id);
                if (dendrite_type_needed == SignalType::Excitatory) {
                    FOR(rep, number_vacant_axons) {
                        need_excit[k].pb(id);
                    }
                }
                else {
                    FOR(rep, number_vacant_axons) {
                        need_inhib[k].pb(id);
                    }
                }
            }
        }
    }
    else {
        //first compute the sums of v for each rank, O(N*R)
        std::vector<double> sum_v_excit(R,0);
        std::vector<double> sum_v_inhib(R,0);
        for (int k=0; k<R; k++) {
            for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {

                sum_v_excit[k] += v[k][neuron_id]*(excitatory_dendrites->get_grown_elements(NeuronID(neuron_id)));

                sum_v_inhib[k] += v[k][neuron_id]*(inhibitory_dendrites->get_grown_elements(NeuronID(neuron_id)));
            }
        }

        for (int k=0; k<R; k++) {//O(R)
            sum_v_excit[k] = fmax(0,sum_v_excit[k]);
            sum_v_inhib[k] = fmax(0,sum_v_inhib[k]);
        }

        //each axon chooses a rank, O(N*R*(nb max axon per neuron))
        for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {

            const NeuronID id{ neuron_id };

            const auto number_vacant_axons = axons->get_free_elements(id);
            if (number_vacant_axons == 0) {
                continue;
            }

            const auto dendrite_type_needed = axons->get_signal_type(id);

            std::vector<double> I(R,0);
            std::vector<int> choices;
            if (dendrite_type_needed == SignalType::Excitatory) {
                for(int k=0; k<R; k++) {
                    I[k] = up[k][neuron_id] * sum_v_excit[k];
                }
                choices = draw_many(I,number_vacant_axons);
                for(const int k : choices) {
                    need_excit[k].push_back(id);
                }
            }//SignalType::Inhibitory
            else {
                for(int k=0; k<R; k++) {
                    I[k] = up[k][neuron_id] * sum_v_inhib[k];
                }
                choices = draw_many(I,number_vacant_axons);
                for(const int k : choices) {
                    need_inhib[k].push_back(id);
                }
            }
        }
    }
    //Communicating with the neurons back
    const auto number_ranks = MPIWrapper::get_num_ranks();
    const auto my_rank = MPIWrapper::get_my_rank();
    //code from the Barnes Hut algo that I don't understand

    //hint at how many rank will access the communication map
    const auto size_hint = std::min(number_neurons, static_cast<number_neurons_type>(number_ranks));

    CommunicationMap<SynapseCreationRequest> synapse_creation_requests_outgoing(number_ranks, size_hint);

    //print the choices
    /*for(int k=0; k<R; k++) {
        choice_rank[k] += need_inhib[k].size()+need_excit[k].size();
    }
    for(int k=0; k<R; k++) {
        std::cout << choice_rank[k] << ",";
    }
    std::cout << std::endl;*/


    //draw excitatory dendrites, O(R*N*log(N))
    for(int k=0; k<R; k++) {
        std::vector<double> I(N,0);
        for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
            I[neuron_id] = vp[k][neuron_id]*(excitatory_dendrites->get_grown_elements(NeuronID(neuron_id)));
        }
        int const nb_need = need_excit[k].size();
        std::vector<int> choices = draw_many(I,nb_need);

        if (!choices.empty()) {
            //need to tell axons in the list to connect to the chosen dendrites
            for(int i=0; i<nb_need; i++) {
                if (need_excit[k][i]!=NeuronID(choices[i])) {
                    SynapseCreationRequest const request = SynapseCreationRequest(NeuronID(choices[i]),need_excit[k][i],SignalType::Excitatory);
                    synapse_creation_requests_outgoing.append(my_rank, request);
                }
            }
        }

    }
    //draw inhibitory dendrites
    for(int k=0; k<R; k++) {
        std::vector<double> I(N,0);
        for (auto neuron_id = 0; neuron_id < number_neurons; ++neuron_id) {
            I[neuron_id] = vp[k][neuron_id]*(inhibitory_dendrites->get_grown_elements(NeuronID(neuron_id)));
        }
        int const nb_need = need_inhib[k].size();
        std::vector<int> choices = draw_many(I,nb_need);

        if (!choices.empty()) {
            //need to tell axons in the list to connect to the chosen dendrites
            for(int i=0; i<nb_need; i++) {
                if (need_inhib[k][i]!=NeuronID(choices[i])) {
                    SynapseCreationRequest request = SynapseCreationRequest(NeuronID(choices[i]),need_inhib[k][i],SignalType::Inhibitory);
                    synapse_creation_requests_outgoing.append(my_rank, request);
                }

            }
        }

    }
    int k;
    synapse_creation_requests_outgoing = MPIWrapper::exchange_requests(synapse_creation_requests_outgoing);

    std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
            result = ForwardConnector::process_requests(synapse_creation_requests_outgoing,excitatory_dendrites, inhibitory_dendrites);
    CommunicationMap<SynapseCreationResponse> responses = result.first;
    PlasticLocalSynapses PLS = result.second.first;
    PlasticDistantInSynapses PDIS = result.second.second;
    PlasticDistantOutSynapses PDOS = ForwardConnector::process_responses(synapse_creation_requests_outgoing,responses,axons);


    return std::make_tuple(PLS,PDIS,PDOS);
}



void LowRankApprox::default_low_rank_approx() {
    u = std::vector<std::vector<double>>(R,std::vector<double>(N,1));
    v = std::vector<std::vector<double>>(R,std::vector<double>(N,1));
    return;
}

bool not_in(int i, std::vector<int> &x) {
    for (int j : x) {
        if(i==j) {
            return false;
        }
    }
    return true;
}

void LowRankApprox::aca_low_rank_approx() {

    u = std::vector<std::vector<double>>(R,std::vector<double>(N,0));
    v = std::vector<std::vector<double>>(R,std::vector<double>(N,0));

    std::vector<int> x(R);
    std::vector<int> y(R+1);
    std::vector<double> d(R);

    y[0] = RandomHolder::get_random_uniform_integer(RandomHolderKey::Algorithm,0,N-1);

    double s=0;
    int maxi=0;
    FOR(k,R) {
        //compute u and find new column
        FOR(i,N) {
            s = Kernel<BarnesHutCell>::Kern(extra_infos->get_position((NeuronID(i))),extra_infos->get_position(NeuronID(y[k])));
            FOR(j,k) {
                s -= u[j][i]*v[j][y[k]]/d[j];
            }
            u[k][i]= s;
        }
        maxi = 0;
        FOR(i,N) {
            if (not_in(i,x)) {
                if ((u[k][i])>(u[k][maxi])) {
                    maxi = i;
                }
            }
        }
        x[k] = maxi;
        d[k] = u[k][maxi];

        //compute v and find new line
        FOR(i,N) {
            s = Kernel<BarnesHutCell>::Kern(extra_infos->get_position(NeuronID(x[k])),extra_infos->get_position(NeuronID(i)));
            FOR(j,k) {
                s -= u[j][x[k]]*v[j][i]/d[j];
            }
            v[k][i]=s;
        }
        maxi = 0;
        FOR(i,N) {
            if (not_in(i,y)) {
                if ((v[k][i])>(v[k][maxi])) {
                    maxi = i;
                }
            }
        }
        y[k+1]=maxi;
    }
    //incorporate d in u
    FOR(k,R) {
        FOR(i,N) {
            u[k][i] /=d[k];
        }
    }
}

void LowRankApprox::centroid_low_rank_approx() {
    int trueR = pow(R,3);
    centroid = std::vector(trueR,std::vector<int>(0));
    u = std::vector<std::vector<double>>(trueR,std::vector<double>(N,0));
    v = std::vector<std::vector<double>>(trueR,std::vector<double>(N,0));
    //allocate neurons to centroids
    int bound = std::ceil(std::cbrt((float)N));
    //O(N)
    FOR(i,N) {
        const auto pos = extra_infos->get_position(NeuronID(i));
        double x = pos.get_x();
        double y = pos.get_y();
        double z = pos.get_z();
        int a = std::floor(x*R/bound);
        int b = std::floor(y*R/bound);
        int c = std::floor(z*R/bound);
        int k = a + b*R + c*R*R;
        centroid[k].pb(i);
        u[k][i] = 1;
        //std::cout << x << ";" << y << ";" << z << "-->" << a << ";" << b << ";" << c << std::endl;
    }
    //O(trueR*N)
    FOR(a,R) {
        FOR(b,R) {
            FOR(c,R) {
                int k = a+b*R+c*R*R;
                double x = (2*((double)a)+1)/(2*R)*bound;
                double y = (2*((double)b)+1)/(2*R)*bound;
                double z = (2*((double)c)+1)/(2*R)*bound;
                //std::cout << x << ";" << y << ";" << z << "-->" << a << ";" << b << ";" << c << std::endl;
                Vec3d p = Vec3d(x,y,z);
                FOR(i,N) {
                    v[k][i] = Kernel<BarnesHutCell>::Kern(p,extra_infos->get_position(NeuronID(i)));
                }
            }
        }
    }
    R = trueR;
}

void LowRankApprox::centroid_weighed_low_rank_approx() {
    int trueR = pow(R,3);
    u = std::vector<std::vector<double>>(trueR,std::vector<double>(N,0));
    v = std::vector<std::vector<double>>(trueR,std::vector<double>(N,0));
    //allocate neurons to centroids
    int bound = std::ceil(std::cbrt((float)N));
    FOR(a,R) {
        FOR(b,R) {
            FOR(c,R) {
                int k = a+b*R+c*R*R;
                double x = (2*((double)a)+1)/(2*R)*bound;
                double y = (2*((double)b)+1)/(2*R)*bound;
                double z = (2*((double)c)+1)/(2*R)*bound;
                //std::cout << x << ";" << y << ";" << z << "-->" << a << ";" << b << ";" << c << std::endl;
                Vec3d p = Vec3d(x,y,z);
                FOR(i,N) {
                    v[k][i] = Kernel<BarnesHutCell>::Kern(p,extra_infos->get_position(NeuronID(i)));
                    u[k][i] = Kernel<BarnesHutCell>::Kern(extra_infos->get_position(NeuronID(i)),p);
                }
            }
        }
    }
    R = trueR;
}

double LowRankApprox::compute_approx(std::vector<std::vector<double>> &u,std::vector<std::vector<double>> &v, int i, int j) {
    double sum = 0;
    FOR(k,R) {
        sum += u[k][i]*v[k][j];
    }
    return sum;
}

fl L1_norm(std::vector<fl> &P, std::vector<fl> &Q) {
    fl sum = 0;
    FOR(i,P.size()) {
        sum += std::abs(P[i] - Q[i]);
    }
    return sum;
}

fl Linf_norm(std::vector<fl> &P, std::vector<fl> &Q) {
    fl sum = 0;
    FOR(i,P.size()) {
        sum = fmax(sum,std::abs(P[i]-Q[i]));
    }
    return sum;
}

std::vector<fl> LowRankApprox::evaluate_error(int rep,bool print_res) {

    fl sum_error_L1 = 0;
    fl sum_error_Linf = 0;
    fl sum_error_L1_pos = 0;
    fl sum_error_Linf_pos = 0;

    FOR(g,rep) {
        int i = RandomHolder::get_random_uniform_integer(RandomHolderKey::Algorithm,0,N-1);
        std::vector<fl> K(N,0);
        std::vector<fl> K_approx(N,0);
        std::vector<fl> K_approx_pos(N,0);
        fl sumK=0,sumK_approx=0,sumK_approx_pos=0;

        FOR(j,N) {
            K[j] = Kernel<BarnesHutCell>::Kern(extra_infos->get_position(NeuronID(i)),extra_infos->get_position(NeuronID(j)));
            sumK += K[j];

            K_approx[j] = compute_approx(u,v,i,j);
            sumK_approx += K_approx[j];

            K_approx_pos[j] = compute_approx(up,vp,i,j);
            sumK_approx_pos += K_approx_pos[j];
        }
        FOR(j,N) {
            K[j] /= sumK;
            K_approx[j] /= sumK_approx;
            K_approx_pos[j] /= sumK_approx_pos;
        }

        sum_error_L1 += L1_norm(K,K_approx);
        sum_error_Linf += Linf_norm(K,K_approx);
        sum_error_L1_pos += L1_norm(K,K_approx_pos);
        sum_error_Linf_pos += Linf_norm(K,K_approx_pos);
    }
    sum_error_L1 /= rep;
    sum_error_Linf /= rep;
    sum_error_L1_pos /= rep;
    sum_error_Linf_pos /= rep;
    if (print_res) {
        LogFiles::print_message_rank(MPIRank::root_rank(), "Before putting to zero coefficients \n L1 error : {} \n Linf error : {}",sum_error_L1,sum_error_Linf);
        LogFiles::print_message_rank(MPIRank::root_rank(), "After putting to zero coefficients \n L1 error : {} \n Linf error : {}",sum_error_L1_pos,sum_error_Linf_pos);
        //LogFiles::print_message_rank(MPIRank::root_rank(), "L1 error : {} \n Linf error : {}",sum_error_L1_pos,sum_error_Linf_pos);
    }


    return {sum_error_L1,sum_error_Linf,sum_error_L1_pos,sum_error_Linf_pos};
}

std::vector<fl> LowRankApprox::evaluate_error_simple(int rep,bool print_res) {

    fl sum_error_L1 = 0;
    fl sum_error_Linf = 0;

    FOR(g, rep) {
        int i = RandomHolder::get_random_uniform_integer(RandomHolderKey::Algorithm, 0, N - 1);
        std::vector<fl> K(N, 0);
        std::vector<fl> K_approx(N, 0);
        fl sumK = 0, sumK_approx = 0;

        FOR(j, N) {
            K[j] = Kernel<BarnesHutCell>::Kern(extra_infos->get_position(NeuronID(i)),
                                               extra_infos->get_position(NeuronID(j)));
            sumK += K[j];

            K_approx[j] = compute_approx(u, v, i, j);
            sumK_approx += K_approx[j];
        }
        FOR(j, N) {
            K[j] /= sumK;
            K_approx[j] /= sumK_approx;
        }

        sum_error_L1 += L1_norm(K, K_approx);
        sum_error_Linf += Linf_norm(K, K_approx);
    }
    sum_error_L1 /= rep;
    sum_error_Linf /= rep;
    if (print_res) {
        LogFiles::print_message_rank(MPIRank::root_rank(),
                                     "Before putting to zero coefficients \n L1 error : {} \n Linf error : {}",
                                     sum_error_L1, sum_error_Linf);
        //LogFiles::print_message_rank(MPIRank::root_rank(), "L1 error : {} \n Linf error : {}",sum_error_L1_pos,sum_error_Linf_pos);
    }


    return {sum_error_L1, sum_error_Linf};
}